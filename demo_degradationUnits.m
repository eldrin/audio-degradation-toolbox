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
filename = 'testdata/472TNA3M_snippet.wav';
[f_audio,samplingFreq]=wavread(filename);

% with default settings:
f_audio_out = degradationUnit_addNoise(f_audio, samplingFreq);

% adjusting some parameters:
parameter.snrRatio = 10; % in dB
parameter.noiseColor = 'pink';  % convenient access to several noise types 
f_audio_out = degradationUnit_addNoise(f_audio, samplingFreq, [], parameter);

wavwrite(f_audio,samplingFreq,16,fullfile(pathOutputDemo,'audio01_in.wav'));
wavwrite(f_audio_out,samplingFreq,16,fullfile(pathOutputDemo,'audio01_out.wav'));


%%
filename = 'testdata/472TNA3M_snippet.wav';
[f_audio,samplingFreq]=wavread(filename);

parameter.snrRatio = 20; % in dB
parameter.loadInternalSound = 1;
parameter.internalSound = 'OldDustyRecording';
f_audio_out = degradationUnit_addSound(f_audio, samplingFreq, [], parameter);

wavwrite(f_audio,samplingFreq,16,fullfile(pathOutputDemo,'audio02_in.wav'));
wavwrite(f_audio_out,samplingFreq,16,fullfile(pathOutputDemo,'audio02_out.wav'));

%%
filename = 'testdata/clarinet.wav';
[f_audio,samplingFreq]=wavread(filename);

parameter.dsFrequency = 8000;
f_audio_out = degradationUnit_applyAliasing(f_audio, samplingFreq, [], parameter);

wavwrite(f_audio,samplingFreq,16,fullfile(pathOutputDemo,'audio03_in.wav'));
wavwrite(f_audio_out,samplingFreq,16,fullfile(pathOutputDemo,'audio03_out.wav'));

%%
filename = 'testdata/472TNA3M_snippet.wav';
[f_audio,samplingFreq]=wavread(filename);

parameter.preNormalization = -5; % 95% quantile will be at this dB level
f_audio_out = degradationUnit_applyClipping(f_audio, samplingFreq, [], parameter);

wavwrite(f_audio,samplingFreq,16,fullfile(pathOutputDemo,'audio04_in.wav'));
wavwrite(f_audio_out,samplingFreq,16,fullfile(pathOutputDemo,'audio04_out.wav'));

%%
filename = 'testdata/472TNA3M_snippet.wav';
[f_audio,samplingFreq]=wavread(filename);

parameter.percentOfSamples = 5;  % signal is scaled so that n% of samples clip  
f_audio_out = degradationUnit_applyClippingAlternative(f_audio, samplingFreq, [], parameter);

wavwrite(f_audio,samplingFreq,16,fullfile(pathOutputDemo,'audio05_in.wav'));
wavwrite(f_audio_out,samplingFreq,16,fullfile(pathOutputDemo,'audio05_out.wav'));

%%
filename = 'testdata/RWC-C08.wav';
[f_audio,samplingFreq]=wavread(filename);

parameter.compressorSlope = 0.9;
parameter.normalizeOutputAudio = 1;
f_audio_out = degradationUnit_applyDynamicRangeCompression(f_audio, samplingFreq, [], parameter);

wavwrite(f_audio,samplingFreq,16,fullfile(pathOutputDemo,'audio06_in.wav'));
wavwrite(f_audio_out,samplingFreq,16,fullfile(pathOutputDemo,'audio06_out.wav'));

%%
filename = 'testdata/clarinet.wav';
[f_audio,samplingFreq]=wavread(filename);

parameter.nApplications = 3;
f_audio_out = degradationUnit_applyHarmonicDistortion(f_audio, samplingFreq, [], parameter);

wavwrite(f_audio,samplingFreq,16,fullfile(pathOutputDemo,'audio07_in.wav'));
wavwrite(f_audio_out,samplingFreq,16,fullfile(pathOutputDemo,'audio07_out.wav'));

%%
filename = 'testdata/p009m_drum.wav';
[f_audio,samplingFreq]=wavread(filename);

parameter.LameOptions = '--preset cbr 128';
f_audio_out = degradationUnit_applyMp3Compression(f_audio, samplingFreq, [], parameter);

wavwrite(f_audio,samplingFreq,16,fullfile(pathOutputDemo,'audio08_in.wav'));
wavwrite(f_audio_out,samplingFreq,16,fullfile(pathOutputDemo,'audio08_out.wav'));

%%
filename = 'testdata/clarinet.wav';
[f_audio,samplingFreq]=wavread(filename);

parameter.changeInPercent = +3;
f_audio_out = degradationUnit_applySpeedup(f_audio, samplingFreq, [], parameter);

wavwrite(f_audio,samplingFreq,16,fullfile(pathOutputDemo,'audio09_in.wav'));
wavwrite(f_audio_out,samplingFreq,16,fullfile(pathOutputDemo,'audio09_out.wav'));

%%
filename = 'testdata/RWC-C08.wav';
[f_audio,samplingFreq]=wavread(filename);

parameter.intensityOfChange = 1.5;
parameter.frequencyOfChange = 0.5;
f_audio_out = degradationUnit_applyWowResampling(f_audio, samplingFreq, [], parameter);

wavwrite(f_audio,samplingFreq,16,fullfile(pathOutputDemo,'audio10_in.wav'));
wavwrite(f_audio_out,samplingFreq,16,fullfile(pathOutputDemo,'audio10_out.wav'));

%%
filename = 'testdata/RWC-C08.wav';
[f_audio,samplingFreq]=wavread(filename);

parameter.stopFrequency = 100;
f_audio_out = degradationUnit_applyHighpassFilter(f_audio, samplingFreq, [], parameter);

wavwrite(f_audio,samplingFreq,16,fullfile(pathOutputDemo,'audio11_in.wav'));
wavwrite(f_audio_out,samplingFreq,16,fullfile(pathOutputDemo,'audio11_out.wav'));


%%
filename = 'testdata/472TNA3M_snippet.wav';
[f_audio,samplingFreq]=wavread(filename);

% Some degradations delay the input signal. If some timepositions are given
% via timepositions_beforeDegr, the corresponding positions will be returned
% in timepositions_afterDegr. 
timepositions_beforeDegr = [2, 3];

% Processing the time positions even works without processing the audio data (f_audio_out is empty afterwards)
parameter.normalizeOutputAudio = 1;
parameter.impulseResponse = [0.5 -0.5 0.5 -0.5 0.5 -0.5 ];
parameter.impulseResponseSampFreq = samplingFreq;
[f_audio_out,timepositions_afterDegr] = degradationUnit_applyImpulseResponse([], [], timepositions_beforeDegr, parameter);

% time positions and audio can also be processed at the same time:
[f_audio_out,timepositions_afterDegr] = degradationUnit_applyImpulseResponse(f_audio, samplingFreq, timepositions_beforeDegr, parameter);
fprintf('degradation_applyFirFilter: adjusting time positions\n');
for k=1:length(timepositions_afterDegr) fprintf('%g -> %g\n',timepositions_beforeDegr(k),timepositions_afterDegr(k)); end

wavwrite(f_audio,samplingFreq,16,fullfile(pathOutputDemo,'audio12_in.wav'));
wavwrite(f_audio_out,samplingFreq,16,fullfile(pathOutputDemo,'audio12_out.wav'));

