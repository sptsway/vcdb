//go:build !arm64 || purego

package engine

// No assembly on this CPU, or the purego tag is set. Use the portable versions.

func l2sq(a, b []float32) float32 { return l2sqGeneric(a, b) }

func dot4(x, c []float32, out *[4]float32) { dot4Generic(x, c, out) }

func l2sq4(q, x []float32, out *[4]float32) { l2sq4Generic(q, x, out) }
