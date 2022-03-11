/*************************************************************************************

Simple Runtime Profiler

Author: Marc Treib

(c) Marc Treib

mailto:treib@in.tum.de

Last Changed: 2013-04-10

*************************************************************************************/
#include "memtrace.h"
#include "profilerstreamwriter.hpp"

#include <iomanip>


ProfilerStreamWriter::ProfilerStreamWriter( std::ostream& stream )
    : m_stream( stream ), m_includeFrameStats( false )
{
    setStreamFormat( );
}

ProfilerStreamWriter::~ProfilerStreamWriter( ) {
}

void ProfilerStreamWriter::started( float timestampMS ) {
    m_stream << "Profiling started at " << timestampMS << " ms\n" << std::endl;
}

void ProfilerStreamWriter::stopped( float timestampMS, float totalProfiledTimeMS, uint totalProfiledFrames, float totalMaxFrameTimeMS ) {
    m_stream << "Profiling stopped at " << timestampMS << " ms";
    if( totalProfiledFrames > 0 ) {
        float totalFps = 1000.0f * totalProfiledFrames / totalProfiledTimeMS;
        m_stream << " (total: " << totalProfiledFrames << " frames in " << totalProfiledTimeMS << " ms = " << totalFps << " fps)";
    } else {
        m_stream << " (total profiled time: " << totalProfiledTimeMS << " ms)";
    }
    m_stream << "\n" << std::endl;
}

void ProfilerStreamWriter::beginOutput( float timestampMS, float profiledTimeMS, uint profiledFrames, float maxFrameTimeMS, float totalProfiledTimeMS, uint totalProfiledFrames, float totalMaxFrameTimeMS ) {
    m_includeFrameStats = ( profiledFrames > 0 );

    m_stream << "Statistics at " << timestampMS << " ms";
    if( m_includeFrameStats ) {
        float fps = 1000.0f * float( profiledFrames ) / profiledTimeMS;
        float mspf = profiledTimeMS / float( profiledFrames );
        float totalFps = 1000.0f * float( totalProfiledFrames ) / totalProfiledTimeMS;
        float totalMspf = totalProfiledTimeMS / float( totalProfiledFrames );
        m_stream << "\n\tThis round: " << std::setw( 5 ) << profiledFrames << " frames"
                 << " in " << std::setw( 8 ) << profiledTimeMS << " ms"
                 << " = " << std::setw( 5 ) << fps << " fps"
                 << ", " << std::setw( 5 ) << mspf << " mspf avg"
                 << ", " << std::setw( 5 ) << maxFrameTimeMS << " mspf max";
        m_stream << "\n\tTotal:     " << std::setw( 6 ) << totalProfiledFrames << " frames"
                 << " in " << std::setw( 8 ) << totalProfiledTimeMS << " ms"
                 << " = " << std::setw( 5 ) << totalFps << " fps"
                 << ", " << std::setw( 5 ) << totalMspf << " mspf avg"
                 << ", " << std::setw( 5 ) << totalMaxFrameTimeMS << " mspf max";
    } else {
        m_stream << " (profiled time: " << profiledTimeMS << " ms; total profiled time: " << totalProfiledTimeMS << " ms)";
    }
    m_stream << "\n";
    
    if( m_includeFrameStats ) {
        m_stream << "                       ms                      :         Count         :\n"
                 << "     Total :  Max/Call : Max/Frame :      Self :     Total : Max/Frame : Name\n"
                 << "---------------------------------------------------------------------------------------------\n";
    } else {
        m_stream << "                 ms                :     Count :\n"
                 << "     Total :  Max/Call :      Self :     Total : Name\n"
                 << "---------------------------------------------------------------------\n";
    }
}

void ProfilerStreamWriter::putSample( const std::string& name, uint parentCount, bool hasUncountedChildren, float totalTimeMS, float maxCallTimeMS, float maxFrameTimeMS, float ownTimeMS, uint totalCount, uint maxFrameCount ) {
    m_stream << " "
             << std::setw( 9 ) << totalTimeMS << " : "
             << std::setw( 9 ) << maxCallTimeMS << " : ";
    if(m_includeFrameStats) {
        m_stream << std::setw( 9 ) << maxFrameTimeMS << " : ";
    }
    m_stream << std::setw( 9 ) << ownTimeMS << " : "
             << std::setw( 9 ) << totalCount << " : ";
    if(m_includeFrameStats) {
        m_stream << std::setw( 9 ) << maxFrameCount << " : ";
    }
    for( uint i = 0; i < parentCount; i++ )
        m_stream << " ";
    m_stream << name;
    if( hasUncountedChildren )
        m_stream << " +";
    m_stream << "\n";
}

void ProfilerStreamWriter::endOutput( ) {
    m_stream << std::endl;
}

void ProfilerStreamWriter::cleared( float timestampMS ) {
    m_stream << "Profiler data cleared at " << timestampMS << " ms\n" << std::endl;
}

void ProfilerStreamWriter::setStreamFormat( ) {
    m_stream << std::showpoint << std::right << std::fixed << std::setprecision( 1 );
}
