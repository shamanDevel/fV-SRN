#include "memtrace.h"
#include <cudaCompress/util/CudaTimer.h>

#include <cassert>


namespace cudaCompress {

namespace util {


CudaTimerResources::CudaTimerResources(bool enabled)
    : m_isEnabled(enabled)
{
}

CudaTimerResources::~CudaTimerResources()
{
    reset();
    for(size_t i = 0; i < m_availableEvents.size(); i++)
    {
        cudaEventDestroy(m_availableEvents[i]);
    }
    m_availableEvents.clear();
    assert(m_activeEvents.empty());
}


const std::map<std::string, float>& CudaTimerResources::getAccumulatedTimes(bool sync)
{
    collectEvents(sync);

    return m_accumulatedTimes;
}

void CudaTimerResources::getAccumulatedTimes(std::vector<std::string>& names, std::vector<float>& times, bool sync)
{
    const std::map<std::string, float>& nameToTime = getAccumulatedTimes(sync);

    for(auto it = nameToTime.cbegin(); it != nameToTime.cend(); ++it)
    {
        names.push_back(it->first);
        times.push_back(it->second);
    }
}


void CudaTimerResources::reset()
{
    if(m_currentSection.m_event != 0)
    {
        m_availableEvents.push_back(m_currentSection.m_event);
    }
    m_currentSection.clear();

    for(size_t i = 0; i < m_activeEvents.size(); i++)
    {
        m_availableEvents.push_back(m_activeEvents[i].m_event);
    }
    m_activeEvents.clear();

    m_accumulatedTimes.clear();
}


void CudaTimerResources::record(const std::string& name)
{
    if(!m_isEnabled) return;

    cudaEvent_t event = getEvent();
    cudaEventRecord(event);

    m_activeEvents.push_back(NamedEvent(name, event));
}


void CudaTimerResources::collectEvents(bool sync)
{
    while(!m_activeEvents.empty())
    {
        // get event
        const NamedEvent& event = m_activeEvents.front();

        // is this an "end" event for the current section?
        if(m_currentSection.m_event != 0)
        {
            if(sync)
            {
                cudaEventSynchronize(event.m_event);
            }
            else if(cudaSuccess != cudaEventQuery(event.m_event))
            {
                // event not done yet, bail out
                break;
            }

            // event is done - record current section time
            float t = 0.0f;
            cudaEventElapsedTime(&t, m_currentSection.m_event, event.m_event);
            m_accumulatedTimes[m_currentSection.m_name] += t;

            // current section event is now available again
            m_availableEvents.push_back(m_currentSection.m_event);
            m_currentSection.clear();
        }

        if(!event.m_name.empty())
        {
            // event is now the new start of the current section
            m_currentSection = event;
        }
        else
        {
            // no name -> not a section start, event is available again
            m_availableEvents.push_back(event.m_event);
        }

        m_activeEvents.pop_front();
    }
}


cudaEvent_t CudaTimerResources::getEvent()
{
    if(m_availableEvents.empty())
    {
        // try to collect finished events
        collectEvents(false);
        // if there's still none available, create a new one
        if(m_availableEvents.empty())
        {
            cudaEvent_t event = 0;
            cudaEventCreate(&event);
            m_availableEvents.push_back(event);
        }
    }

    assert(!m_availableEvents.empty());

    cudaEvent_t result = m_availableEvents.back();
    m_availableEvents.pop_back();
    return result;
}



CudaScopedTimer::CudaScopedTimer(CudaTimerResources& resources)
    : m_resources(resources), m_sectionOpen(false)
{
}

CudaScopedTimer::~CudaScopedTimer()
{
    if(m_sectionOpen)
    {
        (*this)();
    }
}


void CudaScopedTimer::operator() (const std::string& name)
{
    m_resources.record(name);
    m_sectionOpen = !name.empty();
}


}

}
