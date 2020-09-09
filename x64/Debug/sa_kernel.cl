/*****************************************************************************
 * Copyright (c) 2013-2016 Intel Corporation
 * All rights reserved.
 *
 * WARRANTY DISCLAIMER
 *
 * THESE MATERIALS ARE PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL INTEL OR ITS
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THESE
 * MATERIALS, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Intel Corporation is the author of the Materials, and requests that all
 * problem reports or change requests be submitted to it directly
 *****************************************************************************/

#define min(a, b) (((a) < (b)) ? (a) : (b))

__kernel void strConstruct(__global const uchar *input, __global uint *s12, 
    __global uchar *s12_str, __global uint *s0, __global uchar *s0_str, __global const uint *length)
{
    int nthread = get_global_id(0);
    int chunck = *length / 750;
    int maxval = min(*length, (nthread + 1) * chunck);

    for(int i = nthread * chunck; i < maxval; i++)
    {
        if(i % 3 == 0)
        {
            s0[i / 3] = i;
            s0_str[i / 3] = input[i];
        }
        else
        {
            s12[i - (i / 3 + 1)] = i;
            s12_str[i - (i / 3 + 1)] = input[i];
        }
    }
}

__kernel void histogram(__global const uchar *d_suf, __global uint* d_Keys, __global int* d_Histograms,
    const int pass, __local int* loc_histo, int const strLen) 
{
  int it = get_local_id(0);  // i local number of the processor
  int ig = get_global_id(0); // global number = i + g I

  int gr = get_group_id(0); // gr group number

  const int groups = get_num_groups(0);
  int items  = get_local_size(0);

  // initialize the local histograms to zero
  for(int ir = 0; ir < strLen; ir++) {
    loc_histo[ir * items + it] = 0;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // range of keys that are analyzed by the work item
  int n = (1U << 25U);
  int sublist_size  = n/groups/items; // size of the sub-list
  int sublist_start = ig * sublist_size; // beginning of the sub-list

  uint key;
  uint shortkey;
  int k;

  // compute the index
  // the computation depends on the transposition
  for(int j = 0; j < sublist_size; j++) {
    k = j + sublist_start;

    key = d_suf[k];

    // extract the group of _BITS bits of the pass
    // the result is in the range 0.._RADIX-1
	// _BITS = size of _RADIX in bits. So basically they
	// represent both the same. 
    shortkey=(( key >> (pass * 4U)) & (strLen-1));

    // increment the local histogram
    loc_histo[shortkey *  items + it ]++;
  }

  // wait for local histogram to finish
  barrier(CLK_LOCAL_MEM_FENCE);

  // copy the local histogram to the global one
  // in this case the global histo is the group histo.
  for(int ir = 0; ir < strLen; ir++) {
    d_Histograms[items * (ir * groups + gr) + it] = loc_histo[ir * items + it];
  }
}

__kernel void scanhistograms(__global int* histo, __local int* temp, __global int* globsum) 
{
    int it = get_local_id(0);
    int ig = get_global_id(0);
    int decale = 1;
    int n = get_local_size(0) << 1;
    int gr = get_group_id(0);

    // load input into local memory
    // up sweep phase
    temp[(it << 1)]     = histo[(ig << 1)];
    temp[(it << 1) + 1] = histo[(ig << 1) + 1];

    // parallel prefix sum (algorithm of Blelloch 1990)
    // This loop runs log2(n) times
    for (int d = n >> 1; d > 0; d >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (it < d) {
            int ai = decale * ((it << 1) + 1) - 1;
            int bi = decale * ((it << 1) + 2) - 1;
            temp[bi] += temp[ai];
        }
        decale <<= 1;
    }

    // store the last element in the global sum vector
    // (maybe used in the next step for constructing the global scan)
    // clear the last element
    if (it == 0) {
        globsum[gr] = temp[n - 1];
        temp[n - 1] = 0;
    }

    // down sweep phase
    // This loop runs log2(n) times
    for (int d = 1; d < n; d <<= 1){
        decale >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);

        if (it < d){
            int ai = decale*((it << 1) + 1) - 1;
            int bi = decale*((it << 1) + 2) - 1;

            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // write results to device memory
    histo[(ig << 1)]       = temp[(it << 1)];
    histo[(ig << 1) + 1]   = temp[(it << 1) + 1];
}

__kernel void reorder(__global const int* d_inKeys, __global int* d_outKeys, __global int* d_Histograms,
    const int pass, __local  int* loc_histo, int const strLen)
{

	int it = get_local_id(0);  // i local number of the processor
	int ig = get_global_id(0); // global number = i + g I

    int gr = get_group_id(0);				// gr group number
    const int groups = get_num_groups(0);	// G: group count
    int items = get_local_size(0);			// group size

    int n = (1U << 25U);
	int start = ig *(n / groups / items);   // index of first elem this work-item processes
    int size  = n / groups / items;			// count of elements this work-item processes

    // take the histogram in the cache
    for (int ir = 0; ir < strLen; ir++){
        loc_histo[ir * items + it] =
            d_Histograms[items * (ir * groups + gr) + it];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

	int newpos;					// new position of element
	uint key;		// key element
	uint shortkey;	// key element within cache (cache line)
	int k;						// global position within input elements

    for (int j = 0; j < size; j++) {
        k = j + start;
        key = d_inKeys[k];
        shortkey = ((key >> (pass * 4U)) & (strLen - 1));	// shift element to relevant bit positions

        newpos = loc_histo[shortkey * items + it];

        d_outKeys[newpos] = key;

        newpos++;
        loc_histo[shortkey * items + it] = newpos;
    }
}

__kernel void compute_rank(__global uint* d_str, __global const uchar* d_keys_srt, __global uint* d_flag,
    __global bool* result, __global const int* str_length)
{
    int id = get_global_id(0);
    int chunck = *str_length / 750;
    int maxval = min(*str_length, (id + 1) * chunck);
    int k;
    int j;

    for(int i = id * chunck; i < maxval; i++)
    {
        if(id==0) d_flag[id]=1;
        k = d_keys_srt[id];
        j = d_keys_srt[id - 1 % *str_length];
        if(k < *str_length+2 && j < *str_length+2)
        {
            if((d_str[k-1]==d_str[j-1]) && (d_str[k]==d_str[j]) && (d_str[k+1]==d_str[j+1])) 
            {
                d_flag[k] = 0; result[0]=0;
            } else {
                d_flag[i] = 1;
            }
        }
    }
}






















__kernel void compute_lcp(__global const uint* sa, __global const uchar* input, __global uint* lcp, int const strLen)
{
    int id = get_global_id(0);
    int h = 0;
    int chunck = strLen / 750;
    int maxval = min(strLen, (id + 1) * chunck);

    for(int i = id * chunck; i < maxval; i++)
    {
        int isa = sa[sa[( sa[i] >> 8U) % (strLen-1)]];
        if(isa > 1)
        {
            int j = sa[isa - 1 % strLen];
            while(input[i + h] == input[j + h])
                h = h + 1;
            lcp[isa] = h;
            if(h > 0) 
            {
                h = h - 1;
            }
        }
    }
}

int get_inverse(uint* sa, int i)
{
    return i;
}