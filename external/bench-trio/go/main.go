// Command klauspost-adapter exposes klauspost/reedsolomon's encoder to the
// Rust harness as a C archive.
//
// The buffers an encoder reads and writes are registered once at
// construction, so a timed Encode call performs no allocation and no pointer
// staging on the Go side. Registered memory belongs to the caller and must
// outlive the encoder: the harness holds the fixtures for the whole shape and
// drops the encoder first.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"errors"
	"runtime"
	"unsafe"

	"github.com/klauspost/reedsolomon"
)

// encoder is one staged systematic encoder over caller-owned memory.
type encoder struct {
	enc        reedsolomon.Encoder
	dataShards int
	shards     [][]byte
}

var (
	lastError error
	encoders  = map[C.uint64_t]*encoder{}
	nextID    C.uint64_t
)

//export kp_gomaxprocs
func kp_gomaxprocs() C.int {
	return C.int(runtime.GOMAXPROCS(0))
}

//export kp_new
func kp_new(
	dataShards C.int,
	parityShards C.int,
	rowLen C.uintptr_t,
	matrix *C.uint8_t,
	sources **C.uint8_t,
	parity **C.uint8_t,
) C.uint64_t {
	rows := make([][]byte, parityShards)
	flat := unsafe.Slice((*byte)(unsafe.Pointer(matrix)), uintptr(dataShards)*uintptr(parityShards))
	for row := range rows {
		rows[row] = flat[uintptr(row)*uintptr(dataShards) : uintptr(row+1)*uintptr(dataShards)]
	}
	enc, err := reedsolomon.New(
		int(dataShards),
		int(parityShards),
		reedsolomon.WithCustomMatrix(rows),
	)
	if err != nil {
		lastError = err
		return 0
	}

	sourcePointers := unsafe.Slice(sources, dataShards)
	parityPointers := unsafe.Slice(parity, parityShards)
	staged := make([][]byte, 0, int(dataShards+parityShards))
	for _, pointer := range sourcePointers {
		staged = append(staged, unsafe.Slice((*byte)(unsafe.Pointer(pointer)), rowLen))
	}
	for _, pointer := range parityPointers {
		staged = append(staged, unsafe.Slice((*byte)(unsafe.Pointer(pointer)), rowLen))
	}

	nextID++
	encoders[nextID] = &encoder{enc: enc, dataShards: int(dataShards), shards: staged}
	return nextID
}

//export kp_encode
func kp_encode(id C.uint64_t) C.int {
	staged, ok := encoders[id]
	if !ok {
		lastError = errors.New("unknown encoder handle")
		return -1
	}
	if err := staged.enc.Encode(staged.shards); err != nil {
		lastError = err
		return -1
	}
	return 0
}

//export kp_encode_idx
func kp_encode_idx(id C.uint64_t, index C.int, data *C.uint8_t, rowLen C.uintptr_t) C.int {
	staged, ok := encoders[id]
	if !ok {
		lastError = errors.New("unknown encoder handle")
		return -1
	}
	shard := unsafe.Slice((*byte)(unsafe.Pointer(data)), rowLen)
	parity := staged.shards[staged.dataShards:]
	if err := staged.enc.EncodeIdx(shard, int(index), parity); err != nil {
		lastError = err
		return -1
	}
	return 0
}

//export kp_last_error
func kp_last_error() *C.char {
	if lastError == nil {
		return nil
	}
	return C.CString(lastError.Error())
}

//export kp_drop
func kp_drop(id C.uint64_t) {
	delete(encoders, id)
}

func main() {}
