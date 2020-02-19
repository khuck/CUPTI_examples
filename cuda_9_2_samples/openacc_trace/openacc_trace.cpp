/*
 * Copyright 2015 NVIDIA Corporation. All rights reserved
 *
 * Sample CUPTI library for OpenACC data collection.
 */

#include <stdlib.h>
#include <stdio.h>
#include <cupti.h>
#include <cuda.h>
#include <openacc.h>
#include <map>
#include <signal.h>
#include <csignal> // or C++ style alternative

// helper macros

#define CUPTI_CALL(call)                                                \
  do {                                                                  \
    CUptiResult _status = call;                                         \
    if (_status != CUPTI_SUCCESS) {                                     \
      const char *errstr;                                               \
      cuptiGetResultString(_status, &errstr);                           \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
              __FILE__, __LINE__, #call, errstr);                       \
      exit(-1);                                                         \
    }                                                                   \
  } while (0)

#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

static size_t openacc_records = 0;

void finalize(void);
/* make sure our static global object is used before we exit */
struct safemap : public  std::map<uint64_t,uint64_t> {
    virtual ~safemap() {
        finalize();
    }
};

safemap & getprev() {
    static safemap previous;
    return previous;
}

void testTimestamp(uint64_t _start, uint64_t _end, uint32_t context, uint32_t stream) {
    uint64_t start = _start;
    uint64_t end = _end;
    //uint64_t start = _start;
    //uint64_t end = _end;
    uint64_t key = ((uint64_t)(context) << 32) + stream;
    if (getprev().count(key) == 0) {
        getprev()[key] = end;
    } else if (start < getprev()[key]) {
        fflush(stdout);
        fflush(stderr);
        fprintf(stderr, "------- Overlap!\n");
        double nano = (double)(getprev()[key] - start);
        double seconds = nano * 1.0e-9;
        fprintf(stderr, "Relative timestamps: %llu < %llu (%9.2f us) on context %lu stream %lu\n",
            start, getprev()[key], (getprev()[key] - start)/1.e3, context, stream);
        double first = ((double)(start)) * 1.0e-9;
        double second = ((double)(getprev()[key])) * 1.0e-9;
        fprintf(stderr, "Relative: %f < %f (%f seconds) on context %lu stream %lu\n",
            first, second, seconds, context, stream);
        fflush(stderr);
        //raise(SIGSTOP);
    }
    getprev()[key] = end;
}

static void
_printActivity(CUpti_Activity *record, const char * type) {
    CUpti_ActivityOpenAcc *oacc =
           (CUpti_ActivityOpenAcc *)record;

    /* These event types cause overlaps.  Ignore them. */
    uint32_t stream = oacc->cuStreamId;
    if (oacc->eventKind == CUPTI_OPENACC_EVENT_KIND_ENTER_DATA) stream = 0;
    if (oacc->eventKind == CUPTI_OPENACC_EVENT_KIND_CREATE) return;
    if (oacc->eventKind == CUPTI_OPENACC_EVENT_KIND_DELETE) return;
    if (oacc->eventKind == CUPTI_OPENACC_EVENT_KIND_ALLOC) return;
    if (oacc->eventKind == CUPTI_OPENACC_EVENT_KIND_FREE) return;
    if (oacc->eventKind == CUPTI_OPENACC_EVENT_KIND_ENQUEUE_LAUNCH) return;
    if (oacc->eventKind == CUPTI_OPENACC_EVENT_KIND_ENQUEUE_UPLOAD) return;
    if (oacc->eventKind == CUPTI_OPENACC_EVENT_KIND_ENQUEUE_DOWNLOAD) return;
    if (oacc->deviceType != acc_device_nvidia) {
        printf("Error: OpenACC device type is %u, not %u (acc_device_nvidia)\n",
               oacc->deviceType, acc_device_nvidia);
        exit(-1);
    }
    printf("%s, Device: %lu, Context: %lu, Stream: %lu, Event Kind: %2lu, Duration: %9.2fus\n", 
      type, oacc->cuDeviceId, oacc->cuContextId, 
      stream, oacc->eventKind, (oacc->end-oacc->start)/1.e3);
    testTimestamp(oacc->start, oacc->end, oacc->cuContextId, oacc->cuStreamId);

    openacc_records++;
}

static void
printActivity(CUpti_Activity *record)
{
  switch (record->kind) {
      case CUPTI_ACTIVITY_KIND_OPENACC_DATA:
      {
        _printActivity(record, "DATA  ");
        break;
      }
      case CUPTI_ACTIVITY_KIND_OPENACC_LAUNCH:
      {
        _printActivity(record, "LAUNCH");
        break;
      }
      case CUPTI_ACTIVITY_KIND_OPENACC_OTHER:
      {
        _printActivity(record, "OTHER ");
        break;
      }
      default:
          ;
  }
}

// CUPTI buffer handling

void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
  uint8_t *bfr = (uint8_t *) malloc(BUF_SIZE + ALIGN_SIZE);
  if (bfr == NULL) {
    printf("Error: out of memory\n");
    exit(-1);
  }

  *size = BUF_SIZE;
  *buffer = ALIGN_BUFFER(bfr, ALIGN_SIZE);
  *maxNumRecords = 0;
}

void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize)
{
  CUptiResult status;
  CUpti_Activity *record = NULL;

  if (validSize > 0) {
    do {
      status = cuptiActivityGetNextRecord(buffer, validSize, &record);
      if (status == CUPTI_SUCCESS) {
        printActivity(record);
      }
      else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED)
        break;
      else {
        CUPTI_CALL(status);
      }
    } while (1);

    // report any records dropped from the queue
    size_t dropped;
    CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
    if (dropped != 0) {
      printf("Dropped %u activity records\n", (unsigned int) dropped);
    }
  }

  free(buffer);
}

void finalize()
{
  cuptiActivityFlushAll(0);
  printf("Found %llu OpenACC records\n", (long long unsigned) openacc_records);
}

// acc_register_library is defined by the OpenACC tools interface
// and allows to register this library with the OpenACC runtime.

extern "C" void
acc_register_library(void *profRegister, void *profUnregister, void *profLookup)
{
  // once connected to the OpenACC runtime, initialize CUPTI's OpenACC interface
  if (cuptiOpenACCInitialize(profRegister, profUnregister, profLookup) != CUPTI_SUCCESS) {
    printf("Error: Failed to initialize CUPTI OpenACC support\n");
    exit(-1);
  }

  printf("Initialized CUPTI OpenACC\n");

  // only interested in OpenACC activity records
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OPENACC_DATA));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OPENACC_LAUNCH));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OPENACC_OTHER));

  // setup CUPTI buffer handling
  CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));

  // at program exit, flush CUPTI buffers and print results
  atexit(finalize);
}

