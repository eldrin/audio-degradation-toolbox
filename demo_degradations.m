
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Audio Degradation Toolbox
%
% Centre for Digital Music, Queen Mary University of London.
% This file copyright 2013 Sebastian Ewert, Matthias Mauch and QMUL.
%    
% This program is free software; you can redistribute it and/or
% modify it under the terms of the GNU General Public License as
% published by the Free Software Foundation; either version 2 of the
% License, or (at your option) any later version.  See the file
% COPYING included with this distribution for more information.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Note: Some degradations impose a delay/temporal distortion on the input
% data. The last example shows, given time positions before the
% degradation, how the corresponding time positions after the degradation
% can be retrieved

%%

clear

addpath(fullfile(pwd,'AudioDegradationToolbox'));

pathOutputDemo = 'demoOutput/';
if ~exist(pathOutputDemo,'dir'), mkdir(pathOutputDemo); end

%%
filename = 'testdata/p009m_drum.wav';
[f_audio,samplingFreq]=wavread(filename);

f_audio_out = applyDegradation('liveRecording', f_audio, samplingFreq);

wavwrite(f_audio,samplingFreq,16,fullfile(pathOutputDemo,'audio_degradation01_in.wav'));
wavwrite(f_audio_out,samplingFreq,16,fullfile(pathOutputDemo,'audio_degradation01_out.wav'));


%%
filename = 'testdata/p009m_drum.wav';
[f_audio,samplingFreq]=wavread(filename);

f_audio_out = applyDegradation('strongMp3Compression', f_audio, samplingFreq);

wavwrite(f_audio,samplingFreq,16,fullfile(pathOutputDemo,'audio_degradation02_in.wav'));
wavwrite(f_audio_out,samplingFreq,16,fullfile(pathOutputDemo,'audio_degradation02_out.wav'));


%%
filename = 'testdata/RWC-C08.wav';
[f_audio,samplingFreq]=wavread(filename);

f_audio_out = applyDegradation('vinylRecording', f_audio, samplingFreq);

wavwrite(f_audio,samplingFreq,16,fullfile(pathOutputDemo,'audio_degradation03_in.wav'));
wavwrite(f_audio_out,samplingFreq,16,fullfile(pathOutputDemo,'audio_degradation03_out.wav'));


%%
filename = 'testdata/RWC-C08.wav';
[f_audio,samplingFreq]=wavread(filename);

f_audio_out = applyDegradation('radioBroadcast', f_audio, samplingFreq);

wavwrite(f_audio,samplingFreq,16,fullfile(pathOutputDemo,'audio_degradation04_in.wav'));
wavwrite(f_audio_out,samplingFreq,16,fullfile(pathOutputDemo,'audio_degradation04_out.wav'));


%%
filename = 'testdata/RWC-C08.wav';
[f_audio,samplingFreq]=wavread(filename);

f_audio_out = applyDegradation('smartPhoneRecording', f_audio, samplingFreq);

wavwrite(f_audio,samplingFreq,16,fullfile(pathOutputDemo,'audio_degradation05_in.wav'));
wavwrite(f_audio_out,samplingFreq,16,fullfile(pathOutputDemo,'audio_degradation05_out.wav'));


%%
filename = 'testdata/RWC-C08.wav';
[f_audio,samplingFreq]=wavread(filename);

f_audio_out = applyDegradation('smartPhonePlayback', f_audio, samplingFreq);

wavwrite(f_audio,samplingFreq,16,fullfile(pathOutputDemo,'audio_degradation06_in.wav'));
wavwrite(f_audio_out,samplingFreq,16,fullfile(pathOutputDemo,'audio_degradation06_out.wav'));


%%
% Some degradations delay the input signal. If some timepositions are given
% via timepositions_beforeDegr, the corresponding positions will be returned
% in timepositions_afterDegr. In this case, there is no need for f_audio
% and samplingFreq as above but they could be specified too.
timepositions_beforeDegr = [5, 60];
[~,timepositions_afterDegr] = applyDegradation('radioBroadcast', [], [], timepositions_beforeDegr);
fprintf('radioBroadcast: corresponding positions:\n');
for k=1:length(timepositions_beforeDegr) fprintf('%g -> %g\n',timepositions_beforeDegr(k),timepositions_afterDegr(k)); end










