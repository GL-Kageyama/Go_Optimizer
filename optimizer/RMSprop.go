package optimizer

import (
	"fmt"
	"math"
)

type RMSprop struct {
	// Learning rate
	lr float64
	// Decay rate
	decayRate float64
	// Moving average of the squared error
	acc map[string][]float64
}

func newRMSprop(lr, decayRate float64, params map[string][]float64) *RMSprop {
	acc := make(map[string][]float64)
	for key, value := range params {
		acc[key] = make([]float64, len(value))
	}
	return &RMSprop{lr: lr, decayRate: decayRate, acc: acc}
}

func (r *RMSprop) update(params map[string][]float64, grads map[string][]float64) {
	for key := range params {
		for i := 0; i < len(params[key]); i++ {
			// Update weight parameters
			r.acc[key][i] = r.decayRate*r.acc[key][i] + (1-r.decayRate)*grads[key][i]*grads[key][i]
			params[key][i] -= r.lr * grads[key][i] / (math.Sqrt(r.acc[key][i]) + 1e-7)
		}
	}
}

func ProcessingRMSprop(params map[string][]float64, grads map[string][]float64) {

	// Create RMSprop object
	rmsprop := newRMSprop(0.1, 0.99, params)

	// Update parameters
	rmsprop.update(params, grads)

	// Confirm updated parameters
	fmt.Printf("RMSprop: %+v\n", params)
}
