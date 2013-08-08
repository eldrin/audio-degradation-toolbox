function degradation_config = pubEnvironment()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Preset: pubEnvironment
% Date of Revision: 2013-04
% Programmer: Sebastian Ewert
%
% Description:
% - this degradation preset employs
%   * degradation_addSound to add some sounds originating from a real
%     pub environment
%   * adthelper_normalizeAudio to normalize the output audio
%         
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

degradation_config(1).methodname = 'degradationUnit_addSound';
degradation_config(1).parameter.loadInternalSound = 1;
degradation_config(1).parameter.internalSound = 'PubEnvironment1';
degradation_config(1).parameter.snrRatio = 15; % in dB

degradation_config(2).methodname = 'adthelper_normalizeAudio';

end




