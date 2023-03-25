package optimizer

import (
	"fmt"
	"math"
)

type AdaBound struct {
	// Learning rate
	lr float64
	// beta1, beta2
	beta1, beta2 float64
	// Variables used in calculation
	t    float64
	v, m map[string][]float64
}

func newAdaBound(lr, beta1, beta2 float64) *AdaBound {
	return &AdaBound{
		lr:    lr,
		beta1: beta1,
		beta2: beta2,
		t:     0,
		v:     make(map[string][]float64),
		m:     make(map[string][]float64),
	}
}

func (a *AdaBound) update(params map[string][]float64, grads map[string][]float64) {
	a.t++
	for key := range params {
		// Initialize v and m for each parameter
		if _, ok := a.v[key]; !ok {
			a.v[key] = make([]float64, len(params[key]))
			a.m[key] = make([]float64, len(params[key]))
		}
		for i := 0; i < len(params[key]); i++ {
			// Compute moving averages of the gradient and its square
			a.m[key][i] = a.beta1*a.m[key][i] + (1-a.beta1)*grads[key][i]
			a.v[key][i] = a.beta2*a.v[key][i] + (1-a.beta2)*grads[key][i]*grads[key][i]

			// 2. Compute the learning rate
			step_size := a.lr / (1 - math.Pow(a.beta2, a.t))
			lower_bound := step_size * (1 - 1/(a.t*0.01+1))
			upper_bound := step_size * (1 + 1/(a.t*0.01))

			// 3. Update parameters
			params[key][i] -= math.Min(math.Max(lower_bound, a.lr/math.Sqrt(a.v[key][i])), upper_bound) * a.m[key][i]
		}
	}
}

func ProcessingAdaBound(params map[string][]float64, grads map[string][]float64) {

	// Create AdaBound object
	adaBound := newAdaBound(0.001, 0.9, 0.999)

	// Update parameters
	adaBound.update(params, grads)

	// Check updated parameters
	fmt.Printf("AdaBound: %+v\n", params)
}
