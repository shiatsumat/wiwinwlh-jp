# <a name="ffi">FFI</a>

## 目次

* [純粋関数](#pure-functions)
* [貯蔵可能配列](#storable-arrays)
* [関数ポインタ](#function-pointers)

## <a name="pure-functions">純粋関数</a>

Wrapping pure C functions with primitive types is trivial.

```cpp
/* $(CC) -c simple.c -o simple.o */

int example(int a, int b)
{
  return a + b;
}
```

```haskell
-- ghc simple.o simple_ffi.hs -o simple_ffi
{-# LANGUAGE ForeignFunctionInterface #-}

import Foreign.C.Types

foreign import ccall safe "example" example
    :: CInt -> CInt -> CInt

main = print (example 42 27)
```

## <a name="storable-arrays">貯蔵可能配列</a>

There exists a ``Storable`` typeclass that can be used to provide low-level
access to the memory underlying Haskell values. ``Ptr`` objects in Haskell
behave much like C pointers although arithmetic with them is in terms of bytes
only, not the size of the type associated with the pointer ( this differs from
C).

The Prelude defines Storable interfaces for most of the basic types as well as
types in the ``Foreign.C`` library.

```haskell
class Storable a where
  sizeOf :: a -> Int
  alignment :: a -> Int
  peek :: Ptr a -> IO a
  poke :: Ptr a -> a -> IO ()
```

To pass arrays from Haskell to C we can again use Storable Vector and several
unsafe operations to grab a foreign pointer to the underlying data that can be
handed off to C. Once we're in C land, nothing will protect us from doing evil
things to memory!

```cpp
/* $(CC) -c qsort.c -o qsort.o */
void swap(int *a, int *b)
{
    int t = *a;
    *a = *b;
    *b = t;
}

void sort(int *xs, int beg, int end)
{
    if (end > beg + 1) {
        int piv = xs[beg], l = beg + 1, r = end;

        while (l < r) {
            if (xs[l] <= piv) {
                l++;
            } else {
                swap(&xs[l], &xs[--r]);
            }
        }

        swap(&xs[--l], &xs[beg]);
        sort(xs, beg, l);
        sort(xs, r, end);
    }
}
```

```haskell
-- ghc qsort.o ffi.hs -o ffi
{-# LANGUAGE ForeignFunctionInterface #-}

import Foreign.Ptr
import Foreign.C.Types
import Foreign.ForeignPtr
import Foreign.ForeignPtr.Unsafe

import qualified Data.Vector.Storable as V
import qualified Data.Vector.Storable.Mutable as VM

foreign import ccall safe "sort" qsort
    :: Ptr a -> CInt -> CInt -> IO ()

vecPtr :: VM.MVector s CInt -> ForeignPtr CInt
vecPtr = fst . VM.unsafeToForeignPtr0

main :: IO ()
main = do
  let vs = V.fromList ([1,3,5,2,1,2,5,9,6] :: [CInt])
  v <- V.thaw vs
  withForeignPtr (vecPtr v) $ \ptr -> do
    qsort ptr 0 9
  out <- V.freeze v
  print out
```

The names of foreign functions from a C specific header file can qualified.

```haskell
foreign import ccall unsafe "stdlib.h malloc"
    malloc :: CSize -> IO (Ptr a)
```

Prepending the function name with a ``&`` allows us to create a reference to the
function pointer itself.

```haskell
foreign import ccall unsafe "stdlib.h &malloc"
    malloc :: FunPtr a
```

## <a name="function-pointers">関数ポインタ</a>

Using the above FFI functionality, it's trivial to pass C function pointers into
Haskell, but what about the inverse passing a function pointer to a Haskell
function into C using ``foreign import ccall "wrapper"``.

```cpp
#include <stdio.h>

void invoke(void *fn(int))
{
  int n = 42;
  printf("Inside of C, now we'll call Haskell.\n");
  fn(n);
  printf("Back inside of C again.\n");
}
```

```haskell
{-# LANGUAGE ForeignFunctionInterface #-}

import Foreign
import System.IO
import Foreign.C.Types(CInt(..))

foreign import ccall "wrapper"
  makeFunPtr :: (CInt -> IO ()) -> IO (FunPtr (CInt -> IO ()))

foreign import ccall "pointer.c invoke"
  invoke :: FunPtr (CInt -> IO ()) -> IO ()

fn :: CInt -> IO ()
fn n = do
  putStrLn "Hello from Haskell, here's a number passed between runtimes:"
  print n
  hFlush stdout

main :: IO ()
main = do
  fptr <- makeFunPtr fn
  invoke fptr
```

Will yield the following output:

```bash
Inside of C, now we'll call Haskell
Hello from Haskell, here's a number passed between runtimes:
42
Back inside of C again.
```
