import torch


state_dict = torch.load("./intensity_0.pt")
torch.save(state_dict, "./intensity_0_nozip.pt", _use_new_zipfile_serialization=False)