import requests
import os

API_KEY = "s2k-awJuRspBcGliFyEAbaFjzwTDIfRppVAj1mYgRLQn"
RELEASE_ID = "2022-12-27"
DATASET_NAME = "papers"

# 请求数据集文件列表
url = f"https://api.semanticscholar.org/datasets/v1/release/{RELEASE_ID}/dataset/{DATASET_NAME}"
headers = {"x-api-key": API_KEY}

resp = requests.get(url, headers=headers)
resp.raise_for_status()
data = resp.json()

# 创建保存目录
os.makedirs(DATASET_NAME, exist_ok=True)

# 下载每个分区文件
for i, file_url in enumerate(data["files"], start=1):
    print(f"Downloading {i}/{len(data['files'])}: {file_url}")
    filename = os.path.join(DATASET_NAME, f"{DATASET_NAME}_{i}.json.gz")
    if os.path.exists(filename):
        print(f"Skipping {i}/{len(data['files'])} (already exists): {filename}")
        continue
    r = requests.get(file_url, stream=True)
    r.raise_for_status()
    with open(filename, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

print("下载完成！")
