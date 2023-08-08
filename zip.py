import brotli

# 压缩
with open("model_mixd.pkl", "rb") as source, open("model_compressed.brotli", "wb") as dest:
    compressed_data = brotli.compress(source.read())
    dest.write(compressed_data)

# 解压缩
with open("model_compressed.brotli", "rb") as source, open("model_decompressed.pkl", "wb") as dest:
    decompressed_data = brotli.decompress(source.read())
    dest.write(decompressed_data)
