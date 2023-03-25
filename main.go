package main

import (
	"./optimizer"
)

func main() {

	// Initialize parameters
	params := make(map[string][]float64)
	params["weight"] = []float64{0.5, 0.3, -0.1}
	params["bias"] = []float64{0.1}

	// Initialize gradients
	grads := make(map[string][]float64)
	grads["weight"] = []float64{0.0, 0.0, 0.0}
	grads["bias"] = []float64{0.0}

	// SGD
	optimizer.ProcessingSGD(params, grads)
	// AdaGrad
	optimizer.ProcessingAdaGrad(params, grads)
	// AdaDelta
	optimizer.ProcessingAdaDelta(params, grads)
	// RMSprop
	optimizer.ProcessingRMSprop(params, grads)
	// Adam
	optimizer.PorocessingAdam(params, grads)
	// Nadam
	optimizer.ProcessingNadam(params, grads)
	// AMSGrad
	optimizer.ProcessingAMSGrad(params, grads)
	// AdaBound
	optimizer.ProcessingAdaBound(params, grads)
}
