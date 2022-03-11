/*************************************************************************************

Simple Runtime Profiler

Author: Marc Treib

(c) Marc Treib

mailto:treib@in.tum.de

Last Changed: 2013-04-10

*************************************************************************************/

#ifndef _profilerstreamwriter_hpp_
#define _profilerstreamwriter_hpp_


#include <ostream>
#include <string>

#include "profiler.hpp"


// Implementation of IProfilerOutputHandler that writes formatted output to an externally created std::ostream.
class ProfilerStreamWriter : public IProfilerOutputHandler {
public:
    ProfilerStreamWriter( std::ostream& stream );
    virtual ~ProfilerStreamWriter( );

    void started( float timestampMS );
    void stopped( float timestampMS, float totalProfiledTimeMS, uint totalProfiledFrames, float totalMaxFrameTimeMS );

    void beginOutput( float timestampMS, float profiledTimeMS, uint profiledFrames, float maxFrameTimeMS, float totalProfiledTimeMS, uint totalProfiledFrames, float totalMaxFrameTimeMS );
    void putSample( const std::string& name, uint parentCount, bool hasUncountedChildren, float totalTimeMS, float maxCallTimeMS, float maxFrameTimeMS, float ownTimeMS, uint totalCount, uint maxFrameCount );
    void endOutput( );

    void cleared( float timestampMS );

private:
    void setStreamFormat( );

    std::ostream& m_stream;
    bool m_includeFrameStats;
};


#endif // _profilerstreamwriter_hpp_
