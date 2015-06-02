EXP = {'mito_check_test':
        {
        'mapping_files' : {
            'Screen_Plate_01': '/groups/gerlich/members/ChristophSommer/data/mito_check_mappings/0001_02.txt',
            },
        'cellh5_files' : {
            'Screen_Plate_01': '/groups/gerlich/members/ChristophSommer/data/mito_check_ch5_repack/0001_02.ch5',
            },
         
        'rows' : ["%05d" % d for d in range(1,30)],
        'cols' : (1,),
        'training_sites' : (1,),
        }
       }   