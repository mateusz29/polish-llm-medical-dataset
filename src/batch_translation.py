import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def poll_batch_job(job_name, interval=30):
    completed_states = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}

    while True:
        batch_job = client.batches.get(name=job_name)

        if batch_job.state.name in completed_states:
            return batch_job

        print(f"Batch job {job_name} is still running...")
        time.sleep(interval)


def process_batch_job(file_path):
    uploaded_file = client.files.upload(
        file=str(file_path),
        # in docs mime_type="jsonl", but that causes errors
        # fix from: https://github.com/googleapis/python-genai/issues/1590
        config=types.UploadFileConfig(display_name=file_path.stem, mime_type="application/json"),
    )
    print(f"Uploaded file: {uploaded_file.name}")

    batch_job = client.batches.create(
        model="models/gemini-2.0-flash-lite",
        src=uploaded_file.name,
        config={"display_name": f"job-{uploaded_file.name}"},
    )
    print(f"Created batch job: {batch_job.name}")

    batch_job = poll_batch_job(batch_job.name)

    print(f"Job finished with state: {batch_job.state.name}")

    if batch_job.state.name != "JOB_STATE_SUCCEEDED":
        print("Job failed!")
        if batch_job.error:
            print(f"Error: {batch_job.error}")
        return

    if batch_job.dest and batch_job.dest.file_name:
        result_file_name = batch_job.dest.file_name
        print(f"Downloading results: {result_file_name}")
        file_content = client.files.download(file=result_file_name)

        res_dir = file_path.parent / "results"
        res_dir.mkdir(parents=True, exist_ok=True)
        res_path = res_dir / f"{file_path.stem}_results.jsonl"

        with open(res_path, "w", encoding="utf-8") as f:
            f.write(file_content.decode("utf-8"))
    else:
        print("No results found.")


def cleanup():
    # delete batches
    batches = client.batches.list(config={"page_size": 100})

    for b in batches.page:
        print(f"Job Name: {b.name}")
        print(f"  - Display Name: {b.display_name}")
        print(f"  - Create Time: {b.create_time.strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            job_to_delete_name = b.name
            client.batches.delete(name=job_to_delete_name)
            print(f"Deleted {job_to_delete_name}")
        except Exception as e:
            print(f"Error deleting job: {e}")

    # delete files
    files = client.files.list(config={"page_size": 100})

    for b in files.page:
        print(f"File Name: {b.name}")
        print(f"  - Display Name: {b.display_name}")
        print(f"  - Uri: {b.uri}")
        print(f"  - Create Time: {b.create_time.strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            file_to_delete_name = b.name
            client.files.delete(name=file_to_delete_name)
            print(f"Deleted {file_to_delete_name}")
        except Exception as e:
            print(f"Error deleting file: {e}")


def main():
    batch_files = (Path(__file__).parent.parent / "batches").glob("*.jsonl")
    for file in batch_files:
        process_batch_job(file)
        #cleanup()


if __name__ == "__main__":
    main()
