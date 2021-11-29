# Pytorch-Lightning-Helpers
Customization on top of pytorch-lightning for quick and easy experimentations. Inspired by [Speechbrain's](https://github.com/speechbrain/speechbrain) mixed yaml configuration and code customization approach, and [Pytorch-lightning's](https://github.com/PyTorchLightning/pytorch-lightning) modular approach.

## Principles
- Plug-and-playable loss functions, models, data-modules and loggers.
- Configuration-as-code
- Minimal pass by position, use pass by name if possible
- Functional and high composability

## TODO 
- [ ] Implement PFN-like logger (Ref: https://github.com/pfnet/pytorch-pfn-extras)
- [ ] Figure out a way to deal with increasing config file length (Ref: https://github.com/facebookresearch/hydra)
- [ ] Think of a shorter name
