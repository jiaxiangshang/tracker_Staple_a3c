seq={
'vot15_bag'
'vot15_ball1'
'vot15_ball2'
'vot15_basketball'
'vot15_birds1'
'vot15_birds2'
'vot15_blanket'
'vot15_bmx'
'vot15_bolt1'
'vot15_bolt2'
'vot15_book'
'vot15_butterfly'
'vot15_car1'
'vot15_car2'
'vot15_crossing'
'vot15_dinosaur'
'vot15_fernando'
'vot15_fish1'
'vot15_fish2'
'vot15_fish3'
'vot15_fish4'
'vot15_girl'
'vot15_glove'
'vot15_godfather'
'vot15_graduate'
'vot15_gymnastics1'
'vot15_gymnastics2'
'vot15_gymnastics3'
'vot15_gymnastics4'
'vot15_hand'
'vot15_handball1'
'vot15_handball2'
'vot15_helicopter'
'vot15_iceskater1'
'vot15_iceskater2'
'vot15_leaves'
'vot15_marching'
'vot15_matrix'
'vot15_motocross1'
'vot15_motocross2'
'vot15_nature'
'vot15_octopus'
'vot15_pedestrian1'
'vot15_pedestrian2'
'vot15_rabbit'
'vot15_racing'
'vot15_road'
'vot15_shaking'
'vot15_sheep'
'vot15_singer1'
'vot15_singer2'
'vot15_singer3'
'vot15_soccer1'
'vot15_soccer2'
'vot15_soldier'
'vot15_sphere'
'vot15_tiger'
'vot15_traffic'
'vot15_tunnel'
'vot15_wiper'
};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Debug
sequence_list_path = '../../../Data/VOT_2015/list.txt';
fid_list = fopen(sequence_list_path);

seq_list = {};
line = fgetl(fid_list);
while ischar(line)
    seq_list = {seq_list{:} line};
    line = fgetl(fid_list);
end

for s=1:numel(seq)
   ExpRL_runTracker(seq_list{1,s},1);
end

