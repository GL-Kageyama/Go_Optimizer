# Go Optimizer function
## Optimizer

## Formula
### SGD(Stochastic Gradient Descent）
![SGD](https://user-images.githubusercontent.com/36861752/227699799-644b99c6-2cc0-4916-9713-83509eb06e09.png)
### AdaGrad
![AdaGrad](https://user-images.githubusercontent.com/36861752/227699813-07a03909-725b-4f73-a76f-76a59dd5a2f9.png)
### RMSprop
![RMSprop](https://user-images.githubusercontent.com/36861752/227699837-a602d74a-bf4d-4f83-b006-e524d7065350.png)
### AdaDelta
![AdaDelta](https://user-images.githubusercontent.com/36861752/227699853-2e17ff47-5615-42b7-bdf2-263b60ccafd7.png)
### Adam
![Adam](https://user-images.githubusercontent.com/36861752/227699879-8b67cb0a-3d33-4a0d-ac2f-728971b0e1af.png)
### Nadam
![Nadam](https://user-images.githubusercontent.com/36861752/227699889-e680feff-033f-42cd-af3e-3aef08480df0.png)
### AMSGrad
![AMSGrad](https://user-images.githubusercontent.com/36861752/227699911-2aeec1aa-66a2-4a17-aad3-b4914ea34c71.png)
### AdaBound
![AdaBound](https://user-images.githubusercontent.com/36861752/227699926-217ea881-3b5b-4b1e-868a-ffa745cb8cd9.png)

## Code Sample in Adam
```Go
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
```

## Output Sample
~/Go_Optimizer $ go build -o main main.go  
~/Go_Optimizer $ ./main   
SGD: map[bias:[0.1] weight:[0.5 0.3 -0.1]]  
AdaGrad: map[bias:[0.1] weight:[0.5 0.3 -0.1]]  
AdaDelta: map[bias:[0.1] weight:[0.5 0.3 -0.1]]  
RMSprop: map[bias:[0.1] weight:[0.5 0.3 -0.1]]  
Adam: map[bias:[0.1] weight:[0.5 0.3 -0.1]]  
Nadam: map[bias:[0.1] weight:[0.5 0.3 -0.1]]  
AMSGrad: map[bias:[0.1] weight:[0.5 0.3 -0.1]]  
AdaBound: map[bias:[0.1] weight:[0.5 0.3 -0.1]]  
