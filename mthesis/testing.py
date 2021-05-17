import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--test", action="store_true")

args = parser.parse_args()

if args.test: print("store_true -> true")
else: print("store_true -> false")