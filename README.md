# Polish LLM Medical Dataset & Translation Pipeline

A data processing and transaltion pipeline designed to prepare high-quality Polish datasets for LLM fine-tuning. The project focuses on transforming and aligning English medical instruction datasets into Polish while preserving domain-specific terminology and structure.

The pipeline integrates multiple datasets:
- https://huggingface.co/datasets/lavita/medical-qa-datasets 
- https://huggingface.co/datasets/lavita/MedREQAL 
- https://huggingface.co/datasets/lavita/AlpaCare-MedInstruct-52k

## Pipeline Overview

* Data ingestion and merging from multiple sources
* Data cleaning and normalization (handling missing, empty, and inconsistent entries)
* Batch translation using API-driven workflows
* Dataset consolidation into a unified, structured format
