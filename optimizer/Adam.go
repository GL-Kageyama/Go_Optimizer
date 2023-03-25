package optimizer

import "fmt"

type Adam struct {
	// learning rate
	lr float64
	// coefficients for moving averages of gradients (beta1, beta2)
	beta1, beta2 float64
	// dictionary to store moving averages and squared moving averages of parameters
	m, v map[string][]float64
	// parameter update count
	t int
}

func newAdam(lr, beta1, beta2 float64, params map[string][]float64) *Adam {
	m := make(map[string][]float64)
	v := make(map[string][]float64)
	for key, val := range params {
		m[key] = make([]float64, len(val))
		v[key] = make([]float64, len(val))
	}
	return &Adam{lr: lr, beta1: beta1, beta2: beta2, m: m, v: v}
}

func (a *Adam) update(params, grads map[string][]float64) {
	a.t++
	for key := range params {
		// Update m, v
		for i := 0; i < len(params[key]); i++ {
			a.m[key][i] += (1.0 - a.beta1) * (grads[key][i] - a.m[key][i])
			a.v[key][i] += (1.0 - a.beta2) * (grads[key][i]*grads[key][i] - a.v[key][i])
		}
		// Correct m, v
		mb := make([]float64, len(params[key]))
		vb := make([]float64, len(params[key]))
		for i := 0; i < len(params[key]); i++ {
			mb[i] = a.m[key][i] / (1.0 - a.beta1)
			vb[i] = a.v[key][i] / (1.0 - a.beta2)
		}
		// Update parameters
		for i := 0; i < len(params[key]); i++ {
			params[key][i] -= a.lr * mb[i] / (vb[i] + 1e-7)
		}
	}
}

func PorocessingAdam(params map[string][]float64, grads map[string][]float64) {

	// Creating Adam object
	adam := newAdam(0.001, 0.9, 0.999, params)

	// Updating parameters
	adam.update(params, grads)

	// Checking updated parameters
	fmt.Printf("Adam: %+v\n", params)
}
