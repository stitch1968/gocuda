package libraries

import "github.com/stitch1968/gocuda/memory"

func retainLibraryBuffers(buffers ...*memory.Memory) ([]*memory.Retention, error) {
	retentions := make([]*memory.Retention, 0, len(buffers))
	for _, buffer := range buffers {
		retention, err := buffer.Retain()
		if err != nil {
			releaseLibraryBuffers(retentions)
			return nil, err
		}
		retentions = append(retentions, retention)
	}
	return retentions, nil
}

func releaseLibraryBuffers(retentions []*memory.Retention) {
	for _, held := range retentions {
		held.Release()
	}
}
