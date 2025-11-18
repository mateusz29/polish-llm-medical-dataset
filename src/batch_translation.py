import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI

load_dotenv()

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
openai_client = OpenAI()


def poll_gemini_batch_job(job_name, interval=30):
    completed_states = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}

    while True:
        batch_job = gemini_client.batches.get(name=job_name)

        if batch_job.state.name in completed_states:
            return batch_job

        print(f"Batch job {job_name} is still running...")
        time.sleep(interval)


def process_gemini_batch_job(model_name: str, file_path):
    uploaded_file = gemini_client.files.upload(
        file=str(file_path),
        # in docs mime_type="jsonl", but that causes errors
        # fix from: https://github.com/googleapis/python-genai/issues/1590
        config=types.UploadFileConfig(display_name=file_path.stem, mime_type="application/json"),
    )
    print(f"Uploaded file: {uploaded_file.name}")

    batch_job = gemini_client.batches.create(
        model=model_name,
        src=uploaded_file.name,
        config={"display_name": f"job-{uploaded_file.name}"},
    )
    print(f"Created batch job: {batch_job.name}")

    batch_job = poll_gemini_batch_job(batch_job.name)

    print(f"Job finished with state: {batch_job.state.name}")

    if batch_job.state.name != "JOB_STATE_SUCCEEDED":
        print("Job failed!")
        if batch_job.error:
            print(f"Error: {batch_job.error}")
        return

    if batch_job.dest and batch_job.dest.file_name:
        result_file_name = batch_job.dest.file_name
        print(f"Downloading results: {result_file_name}")
        file_content = gemini_client.files.download(file=result_file_name)

        res_dir = file_path.parent / "results"
        res_dir.mkdir(parents=True, exist_ok=True)

        model_name = model_name.split("/")[-1]
        res_path = res_dir / f"{model_name}_{file_path.stem}_results.jsonl"

        with open(res_path, "w", encoding="utf-8") as f:
            f.write(file_content.decode("utf-8"))
    else:
        print("No results found.")


def gemini_cleanup():
    # delete batches
    batches = gemini_client.batches.list(config={"page_size": 100})

    for b in batches.page:
        print(f"Job Name: {b.name}")
        print(f"  - Display Name: {b.display_name}")
        print(f"  - Create Time: {b.create_time.strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            job_to_delete_name = b.name
            gemini_client.batches.delete(name=job_to_delete_name)
            print(f"Deleted {job_to_delete_name}")
        except Exception as e:
            print(f"Error deleting job: {e}")

    # delete files
    files = gemini_client.files.list(config={"page_size": 100})

    for b in files.page:
        print(f"File Name: {b.name}")
        print(f"  - Display Name: {b.display_name}")
        print(f"  - Uri: {b.uri}")
        print(f"  - Create Time: {b.create_time.strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            file_to_delete_name = b.name
            gemini_client.files.delete(name=file_to_delete_name)
            print(f"Deleted {file_to_delete_name}")
        except Exception as e:
            print(f"Error deleting file: {e}")


def gemini_batch_translation(model_name, batch_files):
    for file in batch_files:
        process_gemini_batch_job(model_name, file)
        gemini_cleanup()


def poll_openai_batch_job(job_id, interval=30):
    completed_states = {"failed", "completed", "expired", "cancelled"}

    while True:
        batch = openai_client.batches.retrieve(job_id)

        if batch.status in completed_states:
            return batch

        print(f"Batch job {job_id} is still running...")
        time.sleep(interval)


def process_openai_batch_job(file_path):
    with open(file_path, "rb") as f:
        batch_input_file = openai_client.files.create(file=f, purpose="batch")
    print(f"Uploaded file: {batch_input_file.filename}")

    batch_input_file_id = batch_input_file.id
    batch_job = openai_client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"Created batch job: {batch_job.id}")

    batch_job = poll_openai_batch_job(batch_job.id)

    if batch_job.status != "completed":
        print("Job failed!")
        if batch_job.errors:
            print(f"Error: {batch_job.errors}")
        return

    if batch_job.output_file_id:
        result_file_id = batch_job.output_file_id
        print(f"Downloading results: {result_file_id}")
        file_response = openai_client.files.content(result_file_id)

        res_dir = file_path.parent / "results"
        res_dir.mkdir(parents=True, exist_ok=True)

        res_path = res_dir / f"{file_path.stem}_results.jsonl"

        with open(res_path, "w", encoding="utf-8") as f:
            f.write(file_response.text)
    else:
        print("No results found.")


def openai_cleanup():
    # delete files, endpoint to delete batches doesnt exist
    files = openai_client.files.list()

    for file_object in files.data:
        print(f"File Name: {file_object.filename}")
        print(f"  - Id: {file_object.id}")

        try:
            openai_client.files.delete(file_object.id)
            print(f"Deleted file {file_object.id}")
        except Exception as e:
            print(f"Error deleting file: {e}")


def openai_batch_translation(batch_files):
    for file in batch_files:
        process_openai_batch_job(file)

    #openai_cleanup()


def main():
    batch_files = (Path(__file__).parent.parent / "batches").glob("*.jsonl")

    # gemini_batch_translation("models/gemini-2.5-flash-preview-09-2025", batch_files)
    openai_batch_translation(batch_files)


if __name__ == "__main__":
    main()
