#include <torch/torch.h>
#include <torch/cuda.h>
#include <iostream>

int main() {
	
	torch::Tensor tensor = torch::rand({ 2, 3 });
	std::cout << tensor << std::endl;
	std::cout << torch::cuda::is_available()<<'\n';

}