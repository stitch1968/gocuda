//go:build cuda

package libraries

const nativeTestDelayPTX = `
.version 7.0
.target sm_70
.address_size 64
.visible .entry wait_ticks(.param .u64 ticks) {
	.reg .pred %p;
	.reg .u64 %start, %now, %elapsed, %limit;
	ld.param.u64 %limit, [ticks];
	mov.u64 %start, %clock64;
LOOP:
	mov.u64 %now, %clock64;
	sub.u64 %elapsed, %now, %start;
	setp.lt.u64 %p, %elapsed, %limit;
	@%p bra LOOP;
	ret;
}
`
