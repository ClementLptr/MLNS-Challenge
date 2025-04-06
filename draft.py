import pandas as pd

# with open("submissions/submission-ones.csv", "w") as submission_file:
#     submission_file.write("Id,Prediction\n")
#     for i in range(0, 3497):
#         submission_file.write(f"{i},1\n")

# with open("data/train.txt", "r") as test_file:
#     n_lines = 0
#     lines = test_file.readlines()
#     counter = 0
#     for line in lines:
#         n_lines += 1
#         if line.endswith("1\n"):
#             counter += 1
#     print(counter / n_lines)

# data = pd.read_csv("data/node_information_id_remapped.csv", header="infer")
# data.drop(columns=["node_id"], inplace=True)
# data.insert(0, "node_id", range(0, len(data)))
# data.to_csv("data/node_information_id_remapped.csv", index=False)
# data.columns = ["node_id"] + [f"feature_{i}" for i in range(1, len(data.columns))]
# data.to_csv("data/node_information_id_remapped.csv")
