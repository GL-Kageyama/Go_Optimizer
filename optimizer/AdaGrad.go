package optimizer

import (
	"fmt"
	"math"
)

type AdaGrad struct {
	// learning rate
	lr float64
	// learning rate for each parameter
	h map[string][]float64
	// ε
	delta float64
}

func newAdaGrad(lr, delta float64, params map[string][]float64) *AdaGrad {
	h := make(map[string][]float64)
	for key, p := range params {
		h[key] = make([]float64, len(p))
	}
	return &AdaGrad{lr: lr, delta: delta, h: h}
}

func (a *AdaGrad) update(params map[string][]float64, grads map[string][]float64) {
	for key := range params {
		for i := 0; i < len(params[key]); i++ {
			a.h[key][i] += grads[key][i] * grads[key][i]
			params[key][i] -= a.lr * grads[key][i] / (math.Sqrt(a.h[key][i]) + a.delta)
		}
	}
}

func ProcessingAdaGrad(params map[string][]float64, grads map[string][]float64) {

	// Create AdaGrad object
	adaGrad := newAdaGrad(0.1, 1e-7, params)

	// Update parameters
	adaGrad.update(params, grads)

	// Check parameters
	fmt.Printf("AdaGrad: %+v\n", params)

}
