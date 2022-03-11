/*************************************************************************************

Simple Runtime Profiler

Author: Marc Treib

(c) Marc Treib

mailto:treib@in.tum.de

Last Changed: 2013-04-10

*************************************************************************************/
#include "memtrace.h"
#include "profilerlogwriter.hpp"

#include <iomanip>


ProfilerLogWriter::ProfilerLogWriter( ) : m_pWriter( 0 ) {
}

ProfilerLogWriter::ProfilerLogWriter( const std::string& fileName ) : m_pWriter( 0 ) {
    open( fileName );
}

ProfilerLogWriter::~ProfilerLogWriter( ) {
    close( );
}

bool ProfilerLogWriter::open( const std::string& fileName, bool append ) {
    close( );

    std::ios_base::openmode mode = std::ios_base::out;
    if( append )
        mode |= std::ios_base::app;
    m_file.open( fileName.c_str( ), mode );

    if( !m_file.good( ) )
        return false;

    m_pWriter = new ProfilerStreamWriter( m_file );

    return true;
}

void ProfilerLogWriter::close( ) {
    if( m_pWriter ) {
        delete m_pWriter;
        m_pWriter = 0;
    }
    if( m_file.is_open( ) ) {
        m_file.close( );
    }
}

bool ProfilerLogWriter::isOpen( ) const {
    return ( m_file.is_open( ) && m_pWriter != 0 );
}

bool ProfilerLogWriter::isGood( ) const {
    return ( isOpen( ) && m_file.good( ) );
}

void ProfilerLogWriter::started( float timestampMS ) {
    if( m_pWriter )
        m_pWriter->started( timestampMS );
}

void ProfilerLogWriter::stopped( float timestampMS, float totalProfiledTimeMS, uint totalProfiledFrames, float totalMaxFrameTimeMS ) {
    if( m_pWriter )
        m_pWriter->stopped( timestampMS, totalProfiledTimeMS, totalProfiledFrames, totalMaxFrameTimeMS );
}

void ProfilerLogWriter::beginOutput( float timestampMS, float profiledTimeMS, uint profiledFrames, float maxFrameTimeMS, float totalProfiledTimeMS, uint totalProfiledFrames, float totalMaxFrameTimeMS ) {
    if( m_pWriter )
        m_pWriter->beginOutput( timestampMS, profiledTimeMS, profiledFrames, maxFrameTimeMS, totalProfiledTimeMS, totalProfiledFrames, totalMaxFrameTimeMS );
}

void ProfilerLogWriter::putSample( const std::string& name, uint parentCount, bool hasUncountedChildren, float totalTimeMS, float maxCallTimeMS, float maxFrameTimeMS, float ownTimeMS, uint totalCount, uint maxFrameCount ) {
    if( m_pWriter )
        m_pWriter->putSample( name, parentCount, hasUncountedChildren, totalTimeMS, maxCallTimeMS, maxFrameTimeMS, ownTimeMS, totalCount, maxFrameCount );
}

void ProfilerLogWriter::endOutput( ) {
    if( m_pWriter )
        m_pWriter->endOutput( );
}

void ProfilerLogWriter::cleared( float timestampMS ) {
    if( m_pWriter )
        m_pWriter->cleared( timestampMS );
}
