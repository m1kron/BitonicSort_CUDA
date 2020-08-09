#pragma once

namespace helpers
{
// Returns min or max of val1 and val2 depeding on takeMax param.
__device__ int SelectMinMax(const int val1, const int val2, bool takeMax);

// Returns true if val1 and val2 needs to be swapped to produce 
// increasing/decrasing sequence depending on PRODUCE_INCREASING param
template< typename T, bool PRODUCE_INCREASING >
__device__ bool NeedsToBeSwapped(const T& val1, const T& val2);

// General swap method.
template< typename T >
__device__ void Swap(T& val1, T& val2);

// Check and performs swap if needed between val1 and val2.
__device__ bool PerformSwapIfNeeded(int& val1, int& val2, const bool increasing);

////////////////////////////////////////////////////////////////////////////
//
// INLINES:
//
////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__ int SelectMinMax(const int val1, const int val2, bool takeMax)

{
    // Branchless version commented out - it is slower on Turing:
    // --------------------------------------------
    // const bool comp = (val1 > val2) ^ (!takeMax);
    // return val2 ^ ((val1 ^ val2) & -(comp));
    // -------------------------------------------

    // MNMX version:
    // NOTE: I would like to generate just one MNMX instruction with predicate cond, however I don't
    // know how to force compiler to do it since ptx only have separate min and max functions.
    // Following code generates similar SASS:
    // @!P0 IMNMX R17,R14,R16, PT
    // @P0  IMNMX R17,R14,R16, !PT
    // what I need:
    //      IMNMX R17,R14,R16, P0
    // -----------------------------------------------
    // if ( takeMax )
    //   return SelectMinMax<true>(val1,val2);
    // else
    //   return SelectMinMax<false>(val1,val2);
    // -----------------------------------------------

    return ((val1 > val2) ^ (!takeMax)) ? val1 : val2;
}

////////////////////////////////////////////////////////////////////////////
template< typename T, bool PRODUCE_INCREASING >
__device__ bool NeedsToBeSwapped(const T& val1, const T& val2)
{
  return PRODUCE_INCREASING ? (val1 > val2) : (val1 < val2);
}

////////////////////////////////////////////////////////////////////////////
template< typename T >
__device__ void Swap(T& val1, T& val2)
{
  T temp = val1;
  val1 = val2;
  val2 = temp;
}

////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__ bool PerformSwapIfNeeded(int& val1, int& val2, const bool increasing)
{
  const bool swapNeeded = increasing ? NeedsToBeSwapped<int, true>(val1,val2) : NeedsToBeSwapped<int, false>(val1,val2);

  if( swapNeeded )
    helpers::Swap(val1, val2);
  
  return swapNeeded;
}
}