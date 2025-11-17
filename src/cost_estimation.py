from datasets import load_dataset


def word_count(batch, columns):
    counts = []
    for i in range(len(batch[columns[0]])):
        total = 0
        for col in columns:
            if col in batch and batch[col][i]:
                total += len(str(batch[col][i]).split())
        counts.append(total)
    return {"word_count": counts}


def get_word_count_from_columns(dataset, columns):
    dataset = dataset.map(word_count, batched=True, fn_kwargs={"columns": columns})
    count = sum(dataset["word_count"])
    return count


def get_estimated_cost(
    count,
    words_per_1mil_tokens,
    input_cost_per_1mil_tokens,
    output_cost_per_1mil_tokens,
    text_expansion=1.0,
):
    times = count / words_per_1mil_tokens
    cost_input = times * input_cost_per_1mil_tokens
    cost_output = times * output_cost_per_1mil_tokens * text_expansion
    return cost_input + cost_output


def main():
    dataset1 = load_dataset("lavita/medical-qa-datasets", name="all-processed", split="train")  # 239k rows
    dataset2 = load_dataset("lavita/MedREQAL", split="train")  # 2.79k rows
    dataset3 = load_dataset("lavita/AlpaCare-MedInstruct-52k", split="train")  # 52k rows

    columns = ["instruction", "input", "output"]
    count1 = get_word_count_from_columns(dataset1, columns)
    print(f"Word count in medical-qa-datasets: {count1}")

    columns = ["question", "background", "objective", "conclusion"]
    count2 = get_word_count_from_columns(dataset2, columns)
    print(f"Word count in MedREQAL: {count2}")

    columns = ["instruction", "input", "output"]
    count3 = get_word_count_from_columns(dataset3, columns)
    print(f"Word count in AlpaCare-MedInstruct-52k: {count3}")

    total_count = count1 + count2 + count3
    print(f"Total word count from all datasets: {total_count}")

    cost = get_estimated_cost(total_count, 750000, 0.15, 1.25, 1.25)
    print(f"Estimated cost: {cost}$")


if __name__ == "__main__":
    main()
