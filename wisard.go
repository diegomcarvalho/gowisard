package wisard

import (
	"fmt"
)

// RAMNode represents a single RAM node in WiSARD.
type RAMNode struct {
	memory map[string]int
}

// NewRAMNode creates a new RAM node.
func NewRAMNode() *RAMNode {
	return &RAMNode{memory: make(map[string]int)}
}

// Train stores the sub-pattern for a class.
func (ram *RAMNode) Train(address string) {
	ram.memory[address]++
}

// Predict checks if the sub-pattern was seen during training.
func (ram *RAMNode) Predict(address string) bool {
	return ram.memory[address] > 0
}

// WiSARD represents the WiSARD weightless neural network.
type WiSARD struct {
	ramNodes   [][]*RAMNode // [class][ram]
	inputSize  int
	ramSize    int
	numClasses int
}

// NewWiSARD initializes a WiSARD network.
func NewWiSARD(inputSize, ramSize, numClasses int) *WiSARD {
	numRAMs := inputSize / ramSize
	ramNodes := make([][]*RAMNode, numClasses)
	for c := range numClasses {
		ramNodes[c] = make([]*RAMNode, numRAMs)
		for r := 0; r < numRAMs; r++ {
			ramNodes[c][r] = NewRAMNode()
		}
	}
	return &WiSARD{
		ramNodes:   ramNodes,
		inputSize:  inputSize,
		ramSize:    ramSize,
		numClasses: numClasses,
	}
}

// Train WiSARD with input and class label.
func (w *WiSARD) Train(input []int, class int) {
	for r := 0; r < len(w.ramNodes[class]); r++ {
		start := r * w.ramSize
		end := start + w.ramSize
		address := ""
		for i := start; i < end; i++ {
			address += fmt.Sprintf("%d", input[i])
		}
		w.ramNodes[class][r].Train(address)
	}
}

// Predict returns the class with the highest activation (number of RAM nodes > 0) using goroutines.
// If bleaching > 1, only RAM nodes with count >= bleaching are considered activated.
func (w *WiSARD) Predict(input []int, bleaching ...int) int {
	type activationResult struct {
		class      int
		activation float64
	}
	results := make(chan activationResult, w.numClasses)

	bleach := 1
	if len(bleaching) > 0 {
		bleach = bleaching[0]
	}

	for c := 0; c < w.numClasses; c++ {
		go func(class int) {
			activated := 0
			for r := 0; r < len(w.ramNodes[class]); r++ {
				start := r * w.ramSize
				end := start + w.ramSize
				address := ""
				for i := start; i < end; i++ {
					address += fmt.Sprintf("%d", input[i])
				}
				if w.ramNodes[class][r].memory[address] >= bleach {
					activated++
				}
			}
			results <- activationResult{class: class, activation: float64(activated)}
		}(c)
	}

	maxActivation := -1.0
	predicted := -1
	for i := 0; i < w.numClasses; i++ {
		result := <-results
		if result.activation > maxActivation {
			maxActivation = result.activation
			predicted = result.class
		}
	}
	return predicted
}
