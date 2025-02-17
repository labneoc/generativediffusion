import torch
print()
print(torch.cuda.device_count())
print(torch.cuda.is_available())
print(torch.version.cuda)
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_properties(i).name)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using", device, "device")