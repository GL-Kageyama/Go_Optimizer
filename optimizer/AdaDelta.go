package optimizer

import (
	"fmt"
	"math"
)

type AdaDelta struct {
	// Decay rate
	rho float64
	// Constant for numerical stability
	eps float64
	// Moving average of the squared parameter updates
	Eg map[string][]float64
	// Moving average of the squared gradients
	Ex map[string][]float64
}

func newAdaDelta(rho, eps float64, params map[string][]float64) *AdaDelta {
	Eg := make(map[string][]float64)
	Ex := make(map[string][]float64)
	for key := range params {
		Eg[key] = make([]float64, len(params[key]))
		Ex[key] = make([]float64, len(params[key]))
	}
	return &AdaDelta{
		rho: rho,
		eps: eps,
		Eg:  Eg,
		Ex:  Ex,
	}
}

func (a *AdaDelta) update(params map[string][]float64, grads map[string][]float64) {
	for key := range params {
		for i := 0; i < len(params[key]); i++ {
			// Weight decay
			a.Eg[key][i] = a.rho*a.Eg[key][i] + (1-a.rho)*math.Pow(grads[key][i], 2)
			delta := math.Sqrt(a.Ex[key][i]+a.eps) / math.Sqrt(a.Eg[key][i]+a.eps) * grads[key][i]
			params[key][i] -= delta
			// Calculation of squared weight update moving average
			a.Ex[key][i] = a.rho*a.Ex[key][i] + (1-a.rho)*math.Pow(delta, 2)
		}
	}
}

func ProcessingAdaDelta(params map[string][]float64, grads map[string][]float64) {

	// Creating AdaDelta object
	adadelta := newAdaDelta(0.9, 1e-8, params)

	// Updating parameters
	adadelta.update(params, grads)

	// Checking updated parameters
	fmt.Printf("AdaDelta: %+v\n", params)

}
