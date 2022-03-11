#ifndef __SCAN_GLOBALS_H__
#define __SCAN_GLOBALS_H__


namespace cudaCompress {

const int SCAN_CTA_SIZE = 128;                   /**< Number of threads in a CTA */
const int LOG2_SCAN_CTA_SIZE = 7;                 /**< log_2(CTA_SIZE) */

const int SCAN_ELTS_PER_THREAD = 8;              /**< Number of elements per scan thread */

}


#endif
