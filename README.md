# Lightning tools
Customization on top of pytorch-lightning for quick and easy experimentations. Inspired by 

1. [Speechbrain's](https://github.com/speechbrain/speechbrain) mixed yaml configuration, dependency injection and code customization approach.
2. [Pytorch-lightning's](https://github.com/PyTorchLightning/pytorch-lightning) Modular approach.
3. [AWS Cloudformation's](https://aws.amazon.com/cloudformation/) Infrastucture-as-code

## Principles
- Plug-and-playable loss functions, models, data-modules and loggers.
- Configuration-as-code
- Pass by name, minimal pass by position
- Functional and high composability
- Dependency injection

## TODO
- [ ] Dataloader design is not as plug-and-play as expected
- [ ] Grouped batching based on data length.
- [ ] Think of a shorter name
- [ ] Enable multi-gpu processing
