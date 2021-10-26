import subprocess
from PIL import Image
import torch
import os

def input_image_trim():
    subprocess.call(["python", "main.py", "--mode", "align",
                    "--inp_dir", "static/img/assets/representative/custom/female",
                    "--out_dir", "static/img/assets/representative/celeba_hq/src/female"])

def main_run():
    # torch.cuda.is_available()
    # torch.__version__
    # print(torch.version.cuda)
    subprocess.call(["python", "main.py", "--mode", "sample",
                    '--num_workers', '0', "--num_domains", "2", "--resume_iter", "100000", "--w_hpf", "1",
                    "--checkpoint_dir","static/img/expr/checkpoints/celeba_hq","--result_dir", "static/img/expr/results/celeba_hq",
                    "--src_dir", "static/img/assets/representative/celeba_hq/src", "--ref_dir", "static/img/assets/representative/celeba_hq/ref"])

def result_crop(i, filename):
    image = Image.open("static/img/expr/results/celeba_hq/reference.jpg")

    # 256*8448
    # 사진 자르기(세로로 자를수 있음)
    x=256; y=256; w=256; pick=int(i); h=pick*y
    img_trim = image.crop((x, y*pick, x+w, y+h))

    result = filename + "_result (" + i + ").jpg"
    img_trim.save("static/img/expr/results/celeba_hq/" + result)

    return result
