import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


async def poll_batch_job(job_name, interval=30):
    completed_states = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}

    while True:
        batch_job = client.batches.get(name=job_name)

        if batch_job.state.name in completed_states:
            return batch_job

        print(f"Batch job {job_name} is still running...")
        await asyncio.sleep(interval)


async def process_batch_job(file_path):
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

    batch_job = await poll_batch_job(batch_job.name)

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


async def main():
    batch_files = sorted((Path(__file__).parent.parent / "batches").glob("*.jsonl"))
    async with asyncio.TaskGroup() as tg:
        for fp in batch_files:
            tg.create_task(process_batch_job(fp))


if __name__ == "__main__":
    asyncio.run(main())
