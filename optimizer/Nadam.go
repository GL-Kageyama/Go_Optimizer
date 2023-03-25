package optimizer

import (
	"fmt"
	"math"
)

type Nadam struct {
	// learning rate
	lr float64
	// momentum coefficient
	beta1 float64
	// squared momentum coefficient
	beta2 float64
	// momentum
	m map[string][]float64
	// squared momentum
	v map[string][]float64
}

func newNadam(lr, beta1, beta2 float64) *Nadam {
	return &Nadam{
		lr:    lr,
		beta1: beta1,
		beta2: beta2,
		m:     make(map[string][]float64),
		v:     make(map[string][]float64),
	}
}

func (n *Nadam) update(params map[string][]float64, grads map[string][]float64) {
	// Initialization of m and v for the first call
	if len(n.m) == 0 {
		for key := range params {
			n.m[key] = make([]float64, len(params[key]))
			n.v[key] = make([]float64, len(params[key]))
		}
	}

	// Update equations
	for key := range params {
		for i := 0; i < len(params[key]); i++ {
			// Update m and v
			n.m[key][i] = n.beta1*n.m[key][i] + (1.0-n.beta1)*grads[key][i]
			n.v[key][i] = math.Max(n.beta2*n.v[key][i], math.Abs(grads[key][i]))

			// Momentum correction term
			m_hat := n.m[key][i] / (1.0 - math.Pow(n.beta1, float64(i+1)))
			// Squared average correction term
			v_hat := n.v[key][i] / (1.0 - math.Pow(n.beta2, float64(i+1)))
			// Update the parameter
			params[key][i] -= n.lr * m_hat / (math.Sqrt(v_hat) + 1e-7)
		}
	}
}

func ProcessingNadam(params map[string][]float64, grads map[string][]float64) {

	// Create Nadam object
	nadam := newNadam(0.1, 0.9, 0.999)

	// Update parameters
	nadam.update(params, grads)

	// Confirm updated parameters
	fmt.Printf("Nadam: %+v\n", params)
}
