def main():
    print("Hello from project3!")


if __name__ == "__main__":
    main()


import torch

print(torch.cuda.is_available())
print(torch.version.cuda)
