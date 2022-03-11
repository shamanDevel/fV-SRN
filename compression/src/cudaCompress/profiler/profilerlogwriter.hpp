/*************************************************************************************

Simple Runtime Profiler

Author: Marc Treib

(c) Marc Treib

mailto:treib@in.tum.de

Last Changed: 2013-04-10

*************************************************************************************/

#ifndef _profilerlogwriter_hpp_
#define _profilerlogwriter_hpp_


#include <fstream>
#include <string>

#include "profiler.hpp"
#include "profilerstreamwriter.hpp"


// Implementation of IProfilerOutputHandler that writes formatted output to a log file.
// Uses ProfilerStreamWriter for formatting.
class ProfilerLogWriter : public IProfilerOutputHandler {
public:
    ProfilerLogWriter( );
    ProfilerLogWriter( const std::string& fileName );
    virtual ~ProfilerLogWriter( );

    bool open( const std::string& fileName, bool append = false );
    void close( );
    bool isOpen( ) const;
    bool isGood( ) const;

    void started( float timestampMS );
    void stopped( float timestampMS, float totalProfiledTimeMS, uint totalProfiledFrames, float totalMaxFrameTimeMS );

    void beginOutput( float timestampMS, float profiledTimeMS, uint profiledFrames, float maxFrameTimeMS, float totalProfiledTimeMS, uint totalProfiledFrames, float totalMaxFrameTimeMS );
    void putSample( const std::string& name, uint parentCount, bool hasUncountedChildren, float totalTimeMS, float maxCallTimeMS, float maxFrameTimeMS, float ownTimeMS, uint totalCount, uint maxFrameCount );
    void endOutput( );

    void cleared( float timestampMS );

private:
    std::ofstream m_file;
    ProfilerStreamWriter* m_pWriter;
};


#endif // _profilerlogwriter_hpp_
