package optimizer

import (
	"fmt"
	"math"
)

type AMSGrad struct {
	// learning rate
	lr float64
	// exponential decay rates for the moving averages of gradient and its square
	beta1 float64
	beta2 float64
	// the small value added for numerical stability
	epsilon float64
}

func newAMSGrad(lr, beta1, beta2, epsilon float64) *AMSGrad {
	return &AMSGrad{
		lr:      lr,
		beta1:   beta1,
		beta2:   beta2,
		epsilon: epsilon,
	}
}

func (a *AMSGrad) update(params map[string][]float64, grads map[string][]float64, m, v, vhat map[string][]float64, t int) {
	for key := range params {
		for i := 0; i < len(params[key]); i++ {
			// Get the previous m value at t-1
			mPrev := m[key][i]
			// Get the previous v value at t-1
			vPrev := v[key][i]

			// Compute the previous vhat value at t-1
			vhatPrev := vhat[key][i]

			// Compute the previous vmax value at t-1
			vmaxPrev := math.Max(vhatPrev, vPrev)

			// Update the weight
			params[key][i] -= a.lr / (math.Sqrt(vmaxPrev) + a.epsilon) * mPrev

			// Update m
			m[key][i] = a.beta1*mPrev + (1-a.beta1)*grads[key][i]

			// Update v
			v[key][i] = a.beta2*vPrev + (1-a.beta2)*grads[key][i]*grads[key][i]

			// Update vhat
			vhat[key][i] = math.Max(a.beta2*vhatPrev, v[key][i])
		}
	}
}

func ProcessingAMSGrad(params map[string][]float64, grads map[string][]float64) {

	// Create AMSGrad object
	amsGrad := newAMSGrad(0.001, 0.9, 0.999, 1e-7)

	// Initialize m, v, vhat at the previous time step
	m := make(map[string][]float64)
	v := make(map[string][]float64)
	vhat := make(map[string][]float64)

	for key := range params {
		m[key] = make([]float64, len(params[key]))
		v[key] = make([]float64, len(params[key]))
		vhat[key] = make([]float64, len(params[key]))
	}

	// Update parameters
	t := 1
	amsGrad.update(params, grads, m, v, vhat, t)

	// Check parameters
	fmt.Printf("AMSGrad: %+v\n", params)
}
