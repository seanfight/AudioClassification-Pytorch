# argparse是python的命令行解析标准模块
import argparse
parser = argparse.ArgumentParser(description='enter a numbeir')
parser.add_argument('integers',type=str,help="dd")
args = parser.parse_args()
print(args)

