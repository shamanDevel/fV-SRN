/*************************************************************************************

Simple Runtime Profiler

Author: Marc Treib

(c) Marc Treib

mailto:treib@in.tum.de

Last Changed: 2013-04-10

*************************************************************************************/
#include "memtrace.h"
#include "profiler.hpp"

#include <cassert>
#include <limits>
#include <algorithm>

#ifdef _WIN32
    #include <windows.h>
#endif
#ifdef __linux
    #include <sys/time.h>
#endif


namespace {
    uint64 getTime( ) {
#ifdef _WIN32
        LARGE_INTEGER time;
        QueryPerformanceCounter( &time );
        return time.QuadPart;
#elif defined(__linux)
        timeval tv;
        gettimeofday(&tv, 0);
        return uint64(tv.tv_sec) * uint64(1000000) + uint64(tv.tv_usec);
#else
        #error Not implemented yet on this system!
#endif
    }

    uint64 getTimerFrequency( ) {
#ifdef _WIN32
        LARGE_INTEGER perfFreq;
        QueryPerformanceFrequency(&perfFreq);
        return perfFreq.QuadPart;
#elif defined(__linux)
        return uint64(1000000);
#else
        #error Not implemented yet on this system!
#endif
    }
}

Profiler::Profiler( )
    : m_maxSampleDepth( 8 ),
      m_openSampleIndex( -1 ), m_openSampleUncountedChildrenCount( 0 ), m_openSampleCount( 0 ), m_droppedSampleCount( 0 ),
      m_totalProfiledTime( 0 ), m_totalProfiledFrames( 0 ), m_totalMaxFrameTime( 0 ),
      m_lastOutputProfiledTime( 0 ), m_lastOutputProfiledFrames( 0 ), m_maxFrameTimeSinceLastOutput( 0 ),
      m_running( false ), m_startTime( 0 ), m_frameStartTime( 0 )
{
    m_timerFreq = getTimerFrequency( );

    m_globalStartTime = getTime( );
}

bool Profiler::addOutputHandler( IProfilerOutputHandler* handler ) {
    std::pair< std::set< IProfilerOutputHandler* >::iterator, bool > inserted = m_outputHandlers.insert( handler );
    return inserted.second;
}

bool Profiler::removeOutputHandler( IProfilerOutputHandler* handler ) {
    return m_outputHandlers.erase( handler ) != 0;
}

void Profiler::start( ) {
    assert( !m_running );

    uint64 curTime = getTime( );

    // start
    m_startTime = curTime;
    m_frameStartTime = m_startTime;
    m_running = true;

    // notify output handlers
    float timestampMS = getTimestampMS( curTime );
    std::set< IProfilerOutputHandler* >::const_iterator it, itEnd = m_outputHandlers.end( );
    for( it = m_outputHandlers.begin( ); it != itEnd; it++ ) {
        ( *it )->started( timestampMS );
    }
}

void Profiler::stop( ) {
    assert( m_running );

    uint64 curTime = getTime( );

    // stop
    m_totalProfiledTime += curTime - m_startTime;
    m_running = false;

    // notify output handlers
    float timestampMS = getTimestampMS( curTime );
    float totalProfiledTimeMS = getTimeMS( m_totalProfiledTime );
    float totalMaxFrameTimeMS = getTimeMS( m_totalMaxFrameTime );
    std::set< IProfilerOutputHandler* >::const_iterator it, itEnd = m_outputHandlers.end( );
    for( it = m_outputHandlers.begin( ); it != itEnd; it++ ) {
        ( *it )->stopped( timestampMS, totalProfiledTimeMS, m_totalProfiledFrames, totalMaxFrameTimeMS );
    }
}

void Profiler::startSample( const std::string& sampleName ) {
    // ignore if we're not running
    if( !m_running )
        return;

    // check for simple (direct) recursion
    if( m_openSampleIndex >= 0 && m_samples[ m_openSampleIndex ].name == sampleName ) {
        m_samples[ m_openSampleIndex ].openCount++;
        m_samples[ m_openSampleIndex ].totalCount++;
        m_samples[ m_openSampleIndex ].thisFrameCount++;
        if( m_samples[ m_openSampleIndex ].thisFrameCount > m_samples[ m_openSampleIndex ].maxFrameCount )
            m_samples[ m_openSampleIndex ].maxFrameCount = m_samples[ m_openSampleIndex ].thisFrameCount;
        return;
    }

    // check if maximum sample depth was reached
    if( m_openSampleCount >= m_maxSampleDepth ) {
        m_openSampleUncountedChildrenCount++;
        m_samples[ m_openSampleIndex ].hasUncountedChildren = true;
        return;
    }

    // find the sample
    int storeIndex = -1;
    for( uint i = 0; i < MAX_PROFILER_SAMPLES; i++ ) {
        if( !m_samples[ i ].valid ) {
            if( storeIndex < 0 ) {
                storeIndex = i;
            }
        } else {
            if( m_samples[ i ].parentIndex == m_openSampleIndex && m_samples[ i ].name == sampleName ) {
                // this is the sample we want
                assert( m_samples[ i ].openCount == 0 && "Tried to profile a sample which was already being profiled" );

                m_samples[ i ].totalCount++;
                m_samples[ i ].thisFrameCount++;
                if( m_samples[ i ].thisFrameCount > m_samples[ i ].maxFrameCount )
                    m_samples[ i ].maxFrameCount = m_samples[ i ].thisFrameCount;

                m_samples[ i ].openCount++;
                m_samples[ i ].startTime = getTime( );

                m_openSampleIndex = i;
                m_openSampleCount++;

                return;
            }
        }
    }

    // we haven't found it, so it must be a new sample
    assert( storeIndex >= 0 && "Profiler has run out of sample slots!" );
    // handle case when we run out of slots
    if( storeIndex < 0 ) {
        m_droppedSampleCount++;
        if( m_openSampleIndex >= 0 ) {
            m_samples[ m_openSampleIndex ].hasUncountedChildren = true;
        }
        return;
    }

    // init the new sample and start profiling it
    m_samples[ storeIndex ].valid = true;
    m_samples[ storeIndex ].name = sampleName;
    m_samples[ storeIndex ].parentIndex = m_openSampleIndex;
    m_samples[ storeIndex ].parentCount = m_openSampleCount;

    m_samples[ storeIndex ].hasUncountedChildren = false;

    m_samples[ storeIndex ].totalTime = 0;
    m_samples[ storeIndex ].maxCallTime = 0;
    m_samples[ storeIndex ].childTime = 0;
    m_samples[ storeIndex ].thisFrameTime = 0;
    m_samples[ storeIndex ].maxFrameTime = 0;

    m_samples[ storeIndex ].totalCount = 1;
    m_samples[ storeIndex ].thisFrameCount = 1;
    m_samples[ storeIndex ].maxFrameCount = 1;

    m_samples[ storeIndex ].openCount = 1;
    m_samples[ storeIndex ].startTime = getTime( );

    m_openSampleIndex = storeIndex;
    m_openSampleCount++;
}

void Profiler::endSample( ) {
    // ignore if we're not running
    if( !m_running )
        return;

    assert( m_openSampleIndex >= 0 );
    assert( m_openSampleCount > 0 );
    assert( m_samples[ m_openSampleIndex ].openCount > 0 );

    // handle case when we have run out of slots
    if( m_droppedSampleCount > 0 ) {
        m_droppedSampleCount--;
        return;
    }

    // if the sample has uncounted children, nothing to do here
    if( m_openSampleUncountedChildrenCount > 0 ) {
        m_openSampleUncountedChildrenCount--;
        return;
    }

    // if sample is opened multiple times, also nothing to do
    if( m_samples[ m_openSampleIndex ].openCount > 1 ) {
        m_samples[ m_openSampleIndex ].openCount--;
        return;
    }

    uint64 endTime = getTime( );
    // ok, we're done timing
    m_samples[ m_openSampleIndex ].openCount = 0;
    // calculate the time taken this profile
    uint64 timeTaken = endTime - m_samples[ m_openSampleIndex ].startTime;

    // update this sample
    m_samples[ m_openSampleIndex ].totalTime += timeTaken;
    if( timeTaken > m_samples[ m_openSampleIndex ].maxCallTime )
        m_samples[ m_openSampleIndex ].maxCallTime = timeTaken;
    m_samples[ m_openSampleIndex ].thisFrameTime += timeTaken;
    if( m_samples[ m_openSampleIndex ].thisFrameTime > m_samples[ m_openSampleIndex ].maxFrameTime )
        m_samples[ m_openSampleIndex ].maxFrameTime = m_samples[ m_openSampleIndex ].thisFrameTime;

    // update parent sample
    if( m_samples[ m_openSampleIndex ].parentIndex >= 0 )
        m_samples[ m_samples[ m_openSampleIndex ].parentIndex ].childTime += timeTaken;

    // now, the parent sample is the open sample
    m_openSampleIndex = m_samples[ m_openSampleIndex ].parentIndex;
    m_openSampleCount--;
}

void Profiler::endFrame( ) {
    // ignore if we're not running
    if( !m_running )
        return;

    assert( m_openSampleIndex == -1 );
    assert( m_openSampleCount == 0 );

    ++m_totalProfiledFrames;

    uint64 curTime = getTime( );
    m_maxFrameTimeSinceLastOutput = std::max( m_maxFrameTimeSinceLastOutput, curTime - m_frameStartTime );
    m_totalMaxFrameTime = std::max( m_totalMaxFrameTime, m_maxFrameTimeSinceLastOutput );

    // reset per-frame stats of all samples
    for( uint i = 0; i < MAX_PROFILER_SAMPLES; i++ ) {
        m_samples[ i ].thisFrameTime = 0;
        m_samples[ i ].thisFrameCount = 0;
    }

    m_frameStartTime = curTime;
}

void Profiler::output( ) {
    // there shouldn't be any open samples
    assert( m_openSampleIndex == -1 );
    assert( m_openSampleCount == 0 );

    uint64 curTime = getTime( );

    float timestampMS = getTimestampMS( curTime );
    uint64 totalProfiledTime = m_totalProfiledTime;
    if( m_running ) {
        totalProfiledTime += curTime - m_startTime;
    }
    float totalProfiledTimeMS = getTimeMS( totalProfiledTime );
    uint totalProfiledFrames = m_totalProfiledFrames;
    float totalMaxFrameTimeMS = getTimeMS( m_totalMaxFrameTime );

    uint64 profiledTime = 0;
    if( m_running ) {
        profiledTime = totalProfiledTime - m_lastOutputProfiledTime;
    }
    float profiledTimeMS = getTimeMS( profiledTime );
    uint profiledFrames = m_totalProfiledFrames - m_lastOutputProfiledFrames;
    float maxFrameTimeMS = getTimeMS( m_maxFrameTimeSinceLastOutput );

    beginOutput( timestampMS, profiledTimeMS, profiledFrames, maxFrameTimeMS, totalProfiledTimeMS, totalProfiledFrames, totalMaxFrameTimeMS );
    output( -1 );
    endOutput( );

    m_lastOutputProfiledTime = totalProfiledTime;
    m_lastOutputProfiledFrames = totalProfiledFrames;
    m_maxFrameTimeSinceLastOutput = 0;
}

void Profiler::clear( ) {
    // there shouldn't be any open samples
    assert( m_openSampleIndex == -1 );
    assert( m_openSampleCount == 0 );

    for( uint i = 0; i < MAX_PROFILER_SAMPLES; i++ ) {
        m_samples[ i ].valid = false;
    }

    //TODO what to do with m_startTime if m_running? stop and restart?
    m_totalProfiledTime = 0;
    m_totalProfiledFrames = 0;
    m_totalMaxFrameTime = 0;
    m_lastOutputProfiledTime = 0;
    m_lastOutputProfiledFrames = 0;
    m_maxFrameTimeSinceLastOutput = 0;

    // notify output handlers
    float timestampMS = getTimestampMS( m_startTime );
    std::set< IProfilerOutputHandler* >::const_iterator it, itEnd = m_outputHandlers.end( );
    for( it = m_outputHandlers.begin( ); it != itEnd; it++ ) {
        ( *it )->cleared( timestampMS );
    }
}

float Profiler::getTimeMS( uint64 time ) const {
    return 1000.0f * float( time ) / float( m_timerFreq );
}

float Profiler::getTimestampMS( uint64 time ) const {
    return getTimeMS( time - m_globalStartTime );
}

void Profiler::output( int parentIndex ) const {
    for( uint i = 0; i < MAX_PROFILER_SAMPLES; i++ ) {
        if( m_samples[ i ].valid && m_samples[ i ].parentIndex == parentIndex ) {
            // calculate the time spent on the sample itself (excluding children)
            float sampleTotalTimeMS = getTimeMS( m_samples[ i ].totalTime );
            uint64 sampleOwnTime = m_samples[ i ].totalTime - m_samples[ i ].childTime;
            float sampleOwnTimeMS      = getTimeMS( sampleOwnTime );
            float sampleMaxCallTimeMS  = getTimeMS( m_samples[ i ].maxCallTime );
            float sampleMaxFrameTimeMS = getTimeMS( m_samples[ i ].maxFrameTime );

            // output these values
            putSample( m_samples[ i ].name, m_samples[ i ].parentCount, m_samples[ i ].hasUncountedChildren, sampleTotalTimeMS, sampleMaxCallTimeMS, sampleMaxFrameTimeMS, sampleOwnTimeMS, m_samples[ i ].totalCount, m_samples[ i ].maxFrameCount );

            // recurse on children
            output( i );
        }
    }
}

void Profiler::beginOutput( float timestampMS, float profiledTimeMS, uint profiledFrames, float maxFrameTimeMS, float totalProfiledTimeMS, uint totalProfiledFrames, float totalMaxFrameTimeMS ) const {
    std::set< IProfilerOutputHandler* >::const_iterator it, itEnd = m_outputHandlers.end( );
    for( it = m_outputHandlers.begin( ); it != itEnd; it++ ) {
        ( *it )->beginOutput( timestampMS, profiledTimeMS, profiledFrames, maxFrameTimeMS, totalProfiledTimeMS, totalProfiledFrames, totalMaxFrameTimeMS );
    }
}

void Profiler::putSample( const std::string& name, uint parentCount, bool hasUncountedChildren, float totalTimeMS, float maxCallTimeMS, float maxFrameTimeMS, float ownTimeMS, uint totalCount, uint maxFrameCount ) const {
    std::set< IProfilerOutputHandler* >::const_iterator it, itEnd = m_outputHandlers.end( );
    for( it = m_outputHandlers.begin( ); it != itEnd; it++ ) {
        ( *it )->putSample( name, parentCount, hasUncountedChildren, totalTimeMS, maxCallTimeMS, maxFrameTimeMS, ownTimeMS, totalCount, maxFrameCount );
    }
}

void Profiler::endOutput( ) const {
    std::set< IProfilerOutputHandler* >::const_iterator it, itEnd = m_outputHandlers.end( );
    for( it = m_outputHandlers.begin( ); it != itEnd; it++ ) {
        ( *it )->endOutput( );
    }
}
