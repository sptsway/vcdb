//go:build arm64 && !purego

#include "textflag.h"

// FSUB, FADD and FADDP on four float32 lanes, written as raw machine words.
// Go 1.23's arm64 assembler does not know these mnemonics. Later versions do.
// Arguments follow the ARM manual: destination first, then the two sources.
// The encodings were checked by assembling the mnemonic form with a newer Go
// and comparing the machine code word for word.
#define FSUB4S(vd, vn, vm) WORD $(0x4EA0D400 | ((vm)<<16) | ((vn)<<5) | (vd))
#define FADD4S(vd, vn, vm) WORD $(0x4E20D400 | ((vm)<<16) | ((vn)<<5) | (vd))
#define FADDP4S(vd, vn, vm) WORD $(0x6E20D400 | ((vm)<<16) | ((vn)<<5) | (vd))

// func l2sqNEON(a, b *float32, n int) float32
//
// Four vector accumulators, 16 floats per loop pass. Each pass loads 16
// floats from a and from b, subtracts, squares and adds into the accumulators.
TEXT ·l2sqNEON(SB), NOSPLIT, $0-28
	MOVD a+0(FP), R0
	MOVD b+8(FP), R1
	MOVD n+16(FP), R2
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16

l2loop:
	CBZ  R2, l2done
	VLD1.P 64(R0), [V4.S4, V5.S4, V6.S4, V7.S4]
	VLD1.P 64(R1), [V16.S4, V17.S4, V18.S4, V19.S4]
	FSUB4S(4, 4, 16)
	FSUB4S(5, 5, 17)
	FSUB4S(6, 6, 18)
	FSUB4S(7, 7, 19)
	VFMLA V4.S4, V4.S4, V0.S4
	VFMLA V5.S4, V5.S4, V1.S4
	VFMLA V6.S4, V6.S4, V2.S4
	VFMLA V7.S4, V7.S4, V3.S4
	SUB  $16, R2, R2
	B    l2loop

l2done:
	// Add the four accumulators, then add the four lanes together.
	FADD4S(0, 0, 1)
	FADD4S(2, 2, 3)
	FADD4S(0, 0, 2)
	FADDP4S(0, 0, 0)
	FADDP4S(0, 0, 0)
	FMOVS  F0, ret+24(FP)
	RET

// func dot4NEON(x, c *float32, n int, out *[4]float32)
//
// Dot product of x with four rows of c at once. Each pass loads 8 floats of x
// once and uses them against all four rows. Two accumulators per row.
TEXT ·dot4NEON(SB), NOSPLIT, $0-32
	MOVD x+0(FP), R0
	MOVD c+8(FP), R1
	MOVD n+16(FP), R2
	MOVD out+24(FP), R7
	LSL  $2, R2, R6       // row length in bytes
	ADD  R6, R1, R3       // start of row 1
	ADD  R6, R3, R4       // start of row 2
	ADD  R6, R4, R5       // start of row 3
	VEOR V16.B16, V16.B16, V16.B16
	VEOR V17.B16, V17.B16, V17.B16
	VEOR V18.B16, V18.B16, V18.B16
	VEOR V19.B16, V19.B16, V19.B16
	VEOR V20.B16, V20.B16, V20.B16
	VEOR V21.B16, V21.B16, V21.B16
	VEOR V22.B16, V22.B16, V22.B16
	VEOR V23.B16, V23.B16, V23.B16

dotloop:
	CBZ  R2, dotdone
	VLD1.P 32(R0), [V0.S4, V1.S4]
	VLD1.P 32(R1), [V4.S4, V5.S4]
	VLD1.P 32(R3), [V6.S4, V7.S4]
	VLD1.P 32(R4), [V24.S4, V25.S4]
	VLD1.P 32(R5), [V26.S4, V27.S4]
	VFMLA V0.S4, V4.S4, V16.S4
	VFMLA V1.S4, V5.S4, V17.S4
	VFMLA V0.S4, V6.S4, V18.S4
	VFMLA V1.S4, V7.S4, V19.S4
	VFMLA V0.S4, V24.S4, V20.S4
	VFMLA V1.S4, V25.S4, V21.S4
	VFMLA V0.S4, V26.S4, V22.S4
	VFMLA V1.S4, V27.S4, V23.S4
	SUB  $8, R2, R2
	B    dotloop

dotdone:
	// One vector per row, then pairwise adds fold each into one lane.
	FADD4S(16, 16, 17)
	FADD4S(18, 18, 19)
	FADD4S(20, 20, 21)
	FADD4S(22, 22, 23)
	FADDP4S(28, 16, 18)
	FADDP4S(29, 20, 22)
	FADDP4S(30, 28, 29)
	VST1   [V30.S4], (R7)
	RET

// func l2sq4NEON(q, x *float32, n int, out *[4]float32)
//
// Squared distance from q to four consecutive rows of x. q is loaded once per
// pass and reused for all four rows. Each row keeps the same four accumulators
// and the same final additions as l2sqNEON, so the results are bit-for-bit
// the same as four separate l2sqNEON calls.
TEXT ·l2sq4NEON(SB), NOSPLIT, $0-32
	MOVD q+0(FP), R0
	MOVD x+8(FP), R1
	MOVD n+16(FP), R2
	MOVD out+24(FP), R7
	LSL  $2, R2, R6       // row length in bytes
	ADD  R6, R1, R3       // start of row 1
	ADD  R6, R3, R4       // start of row 2
	ADD  R6, R4, R5       // start of row 3
	VEOR V16.B16, V16.B16, V16.B16
	VEOR V17.B16, V17.B16, V17.B16
	VEOR V18.B16, V18.B16, V18.B16
	VEOR V19.B16, V19.B16, V19.B16
	VEOR V20.B16, V20.B16, V20.B16
	VEOR V21.B16, V21.B16, V21.B16
	VEOR V22.B16, V22.B16, V22.B16
	VEOR V23.B16, V23.B16, V23.B16
	VEOR V24.B16, V24.B16, V24.B16
	VEOR V25.B16, V25.B16, V25.B16
	VEOR V26.B16, V26.B16, V26.B16
	VEOR V27.B16, V27.B16, V27.B16
	VEOR V28.B16, V28.B16, V28.B16
	VEOR V29.B16, V29.B16, V29.B16
	VEOR V30.B16, V30.B16, V30.B16
	VEOR V31.B16, V31.B16, V31.B16

l4loop:
	CBZ  R2, l4done
	VLD1.P 64(R0), [V0.S4, V1.S4, V2.S4, V3.S4]

	VLD1.P 64(R1), [V4.S4, V5.S4, V6.S4, V7.S4]
	FSUB4S(4, 4, 0)
	FSUB4S(5, 5, 1)
	FSUB4S(6, 6, 2)
	FSUB4S(7, 7, 3)
	VFMLA V4.S4, V4.S4, V16.S4
	VFMLA V5.S4, V5.S4, V17.S4
	VFMLA V6.S4, V6.S4, V18.S4
	VFMLA V7.S4, V7.S4, V19.S4

	VLD1.P 64(R3), [V4.S4, V5.S4, V6.S4, V7.S4]
	FSUB4S(4, 4, 0)
	FSUB4S(5, 5, 1)
	FSUB4S(6, 6, 2)
	FSUB4S(7, 7, 3)
	VFMLA V4.S4, V4.S4, V20.S4
	VFMLA V5.S4, V5.S4, V21.S4
	VFMLA V6.S4, V6.S4, V22.S4
	VFMLA V7.S4, V7.S4, V23.S4

	VLD1.P 64(R4), [V4.S4, V5.S4, V6.S4, V7.S4]
	FSUB4S(4, 4, 0)
	FSUB4S(5, 5, 1)
	FSUB4S(6, 6, 2)
	FSUB4S(7, 7, 3)
	VFMLA V4.S4, V4.S4, V24.S4
	VFMLA V5.S4, V5.S4, V25.S4
	VFMLA V6.S4, V6.S4, V26.S4
	VFMLA V7.S4, V7.S4, V27.S4

	VLD1.P 64(R5), [V4.S4, V5.S4, V6.S4, V7.S4]
	FSUB4S(4, 4, 0)
	FSUB4S(5, 5, 1)
	FSUB4S(6, 6, 2)
	FSUB4S(7, 7, 3)
	VFMLA V4.S4, V4.S4, V28.S4
	VFMLA V5.S4, V5.S4, V29.S4
	VFMLA V6.S4, V6.S4, V30.S4
	VFMLA V7.S4, V7.S4, V31.S4

	SUB  $16, R2, R2
	B    l4loop

l4done:
	// Row 0
	FADD4S(16, 16, 17)
	FADD4S(18, 18, 19)
	FADD4S(16, 16, 18)
	FADDP4S(16, 16, 16)
	FADDP4S(16, 16, 16)
	FMOVS  F16, 0(R7)
	// Row 1
	FADD4S(20, 20, 21)
	FADD4S(22, 22, 23)
	FADD4S(20, 20, 22)
	FADDP4S(20, 20, 20)
	FADDP4S(20, 20, 20)
	FMOVS  F20, 4(R7)
	// Row 2
	FADD4S(24, 24, 25)
	FADD4S(26, 26, 27)
	FADD4S(24, 24, 26)
	FADDP4S(24, 24, 24)
	FADDP4S(24, 24, 24)
	FMOVS  F24, 8(R7)
	// Row 3
	FADD4S(28, 28, 29)
	FADD4S(30, 30, 31)
	FADD4S(28, 28, 30)
	FADDP4S(28, 28, 28)
	FADDP4S(28, 28, 28)
	FMOVS  F28, 12(R7)
	RET
