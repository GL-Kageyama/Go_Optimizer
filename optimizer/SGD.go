package optimizer

import "fmt"

type SGD struct {
	// Learning rate
	lr float64
}

func newSGD(lr float64) *SGD {
	return &SGD{lr: lr}
}

func (s *SGD) update(params map[string][]float64, grads map[string][]float64) {
	for key := range params {
		for i := 0; i < len(params[key]); i++ {
			params[key][i] -= s.lr * grads[key][i]
		}
	}
}

func ProcessingSGD(params map[string][]float64, grads map[string][]float64) {

	// Create SGD object
	sgd := newSGD(0.1)

	// Update parameters
	sgd.update(params, grads)

	// Check parameters
	fmt.Printf("SGD: %+v\n", params)
}
