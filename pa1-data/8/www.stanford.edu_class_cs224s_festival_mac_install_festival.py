script for installing festival 1.95 on mac os x 10.5 intel the workarounds are from festival talk mailing list by patty and nathan sakunkoo import os urllib urllib2 re def download url fname print downloading s url webfile urllib urlopen url localfile open fname w localfile write webfile read webfile close localfile close get the tar balls url http festvox org packed festival 1.96 html urllib2 urlopen url read files re findall href gz html for f in files skip multisyn sound files because they are big if f find multisyn 0 continue download url f f os system tar xzf f os system rm f patch macosxaudio leopard uses g++ 4.0 which deprecates some function in this file download http www stanford edu class cs224s festival_mac macosxaudio cc macosxaudio cc os system mv macosxaudio cc speech_tools audio macosxaudio cc patch on math macro something about nan download http www stanford edu class cs224s festival_mac est_math h est_math h os system mv est_math h speech_tools include est_math h os chdir speech_tools os system make os chdir make festival use afplay as the audio player download http www stanford edu class cs224s festival_mac siteinit scm siteinit scm os system mv siteinit scm festival lib siteinit scm os chdir festival os system make os chdir bin os system echo hello world festival tts
