# prepare.py
import os
import pickle

input_file_path = "data/은지말.txt"
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

with open(input_file_path, "r", encoding="utf-8") as f:
    data = f.read()

print(f"총 문자 수: {len(data)}")

# 문자 집합 만들기
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("고유 문자 수:", vocab_size)

# 문자 → 정수 매핑
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

# 변환 함수
def encode(s): return [stoi[c] for c in s]

# 정수로 인코딩
train_ids = encode(data)

# 저장
with open(os.path.join(output_dir, "은지말_vocab.pkl"), "wb") as f:
    pickle.dump((stoi, itos), f)

with open(os.path.join(output_dir, "은지말.bin"), "wb") as f:
    f.write(bytearray(train_ids))

print("✅ 변환 완료: data/은지말.bin, data/은지말_vocab.pkl 생성됨")


