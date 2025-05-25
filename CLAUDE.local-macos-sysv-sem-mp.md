# macOS SysV Semaphore Support for Multiprocessing

## Background

The multiprocessing module's semaphore implementation on macOS has issues with `sem_getvalue()` which doesn't work correctly on that platform. This is currently handled by the `HAVE_BROKEN_SEM_GETVALUE` macro, which causes `_get_value()` to raise `NotImplementedError` and `_is_zero()` to use a workaround with `sem_trywait()`.

## Current Implementation Overview

### Key Files
- `Modules/_multiprocessing/semaphore.c` - Main semaphore implementation
- `Modules/_multiprocessing/multiprocessing.h` - Type definitions and platform-specific includes
- `configure.ac` - Contains the test for broken `sem_getvalue`
- `pyconfig.h.in` - Contains `HAVE_BROKEN_SEM_GETVALUE` definition

### Current Platform Support
1. **Windows**: Uses Windows semaphore APIs (`CreateSemaphore`, `WaitForSingleObject`, etc.)
2. **Unix/Linux**: Uses POSIX semaphores (`sem_open`, `sem_wait`, etc.)
3. **macOS**: Uses POSIX semaphores but with `HAVE_BROKEN_SEM_GETVALUE` workaround

### Key Data Structure
```c
typedef struct {
    PyObject_HEAD
    SEM_HANDLE handle;
    unsigned long last_tid;
    int count;
    int maxvalue;
    int kind;
    char *name;
} SemLockObject;
```

### Semaphore Types
- `RECURSIVE_MUTEX` - Can be acquired multiple times by the same thread
- `SEMAPHORE` - Standard counting semaphore

## Plan for Adding macOS SysV Semaphore Support

### 1. Conditional Compilation Structure

Add `#ifdef __APPLE__` blocks to use SysV semaphores while keeping existing code for other platforms.

### 2. Header Changes (multiprocessing.h)

```c
#ifdef __APPLE__
#  include <sys/sem.h>
#  include <sys/ipc.h>
#  include <sys/stat.h>
#  define HAVE_MP_SEMAPHORE
typedef struct {
    int semid;      // SysV semaphore set ID
    int semnum;     // Semaphore number within the set (always 0 for single semaphore)
    key_t key;      // IPC key for the semaphore
    char *name;     // Original name for cleanup
} sysv_sem_t;
typedef sysv_sem_t *SEM_HANDLE;
#endif
```

### 3. macOS-Specific Macro Definitions

```c
#ifdef __APPLE__
#define SEM_CLEAR_ERROR()
#define SEM_GET_LAST_ERROR() errno
#define SEM_FAILED ((SEM_HANDLE)-1)
#define SEM_CREATE(name, val, max) _sysv_sem_create(name, val, max)
#define SEM_CLOSE(sem) _sysv_sem_close(sem)
#define SEM_GETVALUE(sem, pval) _sysv_sem_getvalue(sem, pval)
#define SEM_UNLINK(name) _sysv_sem_unlink(name)
#endif
```

### 4. SysV Semaphore Helper Functions

#### Create Function
```c
static SEM_HANDLE _sysv_sem_create(const char *name, int value, int maxvalue)
{
    sysv_sem_t *sem = PyMem_Malloc(sizeof(sysv_sem_t));
    if (!sem) {
        errno = ENOMEM;
        return SEM_FAILED;
    }
    
    // Generate unique key from name
    // Option 1: Use hash of name
    // Option 2: Create temp file and use ftok()
    key_t key = _name_to_key(name);
    
    // Create semaphore set with one semaphore
    int semid = semget(key, 1, IPC_CREAT | IPC_EXCL | 0600);
    if (semid == -1) {
        PyMem_Free(sem);
        return SEM_FAILED;
    }
    
    // Initialize semaphore value
    union semun {
        int val;
        struct semid_ds *buf;
        unsigned short *array;
    } arg;
    arg.val = value;
    
    if (semctl(semid, 0, SETVAL, arg) == -1) {
        semctl(semid, 0, IPC_RMID);
        PyMem_Free(sem);
        return SEM_FAILED;
    }
    
    sem->semid = semid;
    sem->semnum = 0;
    sem->key = key;
    sem->name = PyMem_Malloc(strlen(name) + 1);
    if (sem->name) {
        strcpy(sem->name, name);
    }
    
    return sem;
}
```

#### Close Function
```c
static int _sysv_sem_close(SEM_HANDLE sem)
{
    if (sem && sem != SEM_FAILED) {
        PyMem_Free(sem->name);
        PyMem_Free(sem);
    }
    return 0;
}
```

#### Get Value Function
```c
static int _sysv_sem_getvalue(SEM_HANDLE sem, int *value)
{
    *value = semctl(sem->semid, sem->semnum, GETVAL);
    return (*value == -1) ? -1 : 0;
}
```

#### Unlink Function
```c
static int _sysv_sem_unlink(const char *name)
{
    key_t key = _name_to_key(name);
    int semid = semget(key, 0, 0);
    if (semid != -1) {
        return semctl(semid, 0, IPC_RMID);
    }
    return 0;
}
```

### 5. Semaphore Operations

#### Wait Operations
```c
#ifdef __APPLE__
static int _sysv_sem_wait(SEM_HANDLE sem)
{
    struct sembuf op = {sem->semnum, -1, 0};
    return semop(sem->semid, &op, 1);
}

static int _sysv_sem_trywait(SEM_HANDLE sem)
{
    struct sembuf op = {sem->semnum, -1, IPC_NOWAIT};
    return semop(sem->semid, &op, 1);
}

static int _sysv_sem_timedwait(SEM_HANDLE sem, struct timespec *deadline)
{
    // macOS might not have semtimedop, so implement polling
    struct timeval now;
    struct sembuf op = {sem->semnum, -1, IPC_NOWAIT};
    
    while (1) {
        if (semop(sem->semid, &op, 1) == 0) {
            return 0;
        }
        if (errno != EAGAIN) {
            return -1;
        }
        
        // Check timeout
        if (gettimeofday(&now, NULL) < 0) {
            return -1;
        }
        if (now.tv_sec > deadline->tv_sec ||
            (now.tv_sec == deadline->tv_sec && 
             now.tv_usec >= deadline->tv_nsec / 1000)) {
            errno = ETIMEDOUT;
            return -1;
        }
        
        // Sleep briefly before retrying
        usleep(1000); // 1ms
    }
}
#endif
```

#### Post Operation
```c
#ifdef __APPLE__
static int _sysv_sem_post(SEM_HANDLE sem)
{
    struct sembuf op = {sem->semnum, 1, 0};
    return semop(sem->semid, &op, 1);
}
#endif
```

### 6. Integration Points

#### Modify acquire() implementation
Replace `sem_wait`, `sem_trywait`, and `sem_timedwait` calls with the SysV equivalents on macOS.

#### Modify release() implementation  
Replace `sem_post` calls with `_sysv_sem_post` on macOS.

#### Modify _rebuild() for serialization
Handle the SysV semaphore structure when rebuilding from pickled data.

### 7. Name-to-Key Mapping Strategy

```c
static key_t _name_to_key(const char *name)
{
    // Option 1: Simple hash
    key_t key = 0;
    for (const char *p = name; *p; p++) {
        key = key * 31 + *p;
    }
    return key;
    
    // Option 2: Use ftok with a temp file
    // char path[PATH_MAX];
    // snprintf(path, sizeof(path), "/tmp/.pymp_%s", name);
    // int fd = open(path, O_CREAT | O_WRONLY, 0600);
    // if (fd != -1) {
    //     close(fd);
    //     key = ftok(path, 1);
    // }
}
```

### 8. Configure Changes

Add to `configure.ac`:
```bash
# Check for SysV semaphore functions on macOS
case $host_os in
  darwin*)
    AC_CHECK_FUNCS([semget semctl semop])
    ;;
esac
```

### 9. Testing Considerations

- All existing multiprocessing semaphore tests should pass
- The `_get_value()` method will actually work on macOS
- Semaphore cleanup on process termination needs testing
- Test with multiple processes using the same named semaphore

### 10. Potential Issues and Solutions

1. **Semaphore Limits**: SysV semaphores have system-wide limits
   - Solution: Document limits, handle ENOSPC errors gracefully

2. **Cleanup**: SysV semaphores persist after process death
   - Solution: Implement proper cleanup in `__del__` and signal handlers

3. **Name Collisions**: Simple hash might cause collisions
   - Solution: Use better hash or ftok-based approach

4. **Security**: SysV semaphores use numeric permissions
   - Solution: Always create with 0600 permissions

## Implementation Notes

- The `HAVE_BROKEN_SEM_GETVALUE` workaround will no longer be needed for macOS
- This change should be transparent to Python code using multiprocessing
- Consider adding a fallback to POSIX semaphores if SysV is not available
- The implementation should handle both named and unnamed semaphores

## Testing the Implementation

1. Run existing tests: `./python -m test test_multiprocessing_fork test_multiprocessing_spawn`
2. Test semaphore value operations specifically
3. Test cleanup after abnormal termination
4. Test with high concurrency scenarios