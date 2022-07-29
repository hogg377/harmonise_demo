from turtle import end_fill


class MenuLog:
    def __init__(self, session_id) -> None:
        self.session_id = session_id
        #the log uses this so it knows which config the results are stored against
        self.config_run_order = []
        # self.user_details = { 'session_id' : '',
        #                         'infosheet4': {'agreed': False},
        #                         'infosheet5': { 'name' : '',
        #                                         'data': '', 
        #                                         'agreed': False},
        #                         'details' : {'native_english_speaker': False,
        #                                     'normal vision': False,
        #                                     'colour deficiency': False,
        #                                     'signature' : '',
        #                                     'dob': '',
        #                                     'sex' : '',
        #                                     'data' : ''},
        #                         'finalconsent': {'agreed' : False}
        #                         }
        self.user_details = { #'name' : '',
                        #'date': '',
                        #'name60' : '',
                        #'date60' : '',
                        'english' : '',
                        'colour': '',
                        'birth' : '',
                        'sex' : '',
                        'vision': '',
                        'name' : ''
                        # consent is handled in paper formating following feedback from beta trials
                        # 'consent14': False,
                        #'final_consent': False
                        }
        self.post_test_questions = {}
        # for i in range(0,18):
        #     self.post_test_questions[f'config_ex_{i+1}'] = {'time' : '', 'engaged' : '', 'part_of_team' : ''}

    # def save_infosheet4(self, menu):
    #     self.user_details['infosheet4'] = True
    #     return

    
    # def save_infosheet5(self, menu):
    #     data = menu.get_input_data()

    
    # def save_details_menu(self, menu):
    #     data = menu.get_input_data()
    #     self.user_details['details'] = data

    # def save_post_test_questions_m1(self, menu):
    #     data = menu.get_input_data()
    #     'time'

    
    # def save_post_test_questions_m2(self, menu):
    #     data = menu.get_input_data()
    #     'engaged'
    #     'part_of_team'
    

    def save_responses(self, menu, menu_id, current_config='none'):
        """
        This function reads the menu_id and then saves the relevant content from the menu to the MenuLog's internal data structures
        """

        data = menu.get_input_data()

        if menu_id == 0:
            self.pickleLog('logs/' + self.session_id)

        if menu_id == 11:
            pass
        elif menu_id == 12:
            pass

        elif menu_id == 13:
            pass

        #Menu 14 is no longer included following feedback from beta trials
        elif menu_id == 14:
            #self.user_details['name'] = data['name']
            #self.user_details['date'] = data['date']
            #if this is called to store details then consent has been given
            #   default consent is falue and confirmation of consent is by clicking the "consent and continue" button which triggers this storage
            #self.user_details['consent14'] = True
            pass

        elif menu_id == 60:
            self.user_details['english'] = data['english']
            self.user_details['vision'] = data['vision']
            self.user_details['colour'] = data['colour']
            #self.user_details['name60'] = data['name']
            self.user_details['birth'] = data['birth']
            self.user_details['sex'] = data['sex']
            self.user_details['name'] = data['name']
            #self.user_details['date60'] = data['date']
            pass

        # Menu 80 is no longer included in the latest version
        # elif menu_id == 80:
        #     #if this is called to store details then consent has been given
        #     #   default consent is falue and confirmation of consent is by clicking the "consent and continue" button which triggers this storage
        #     self.user_details['final_consent'] = True
        #     pass

        elif (menu_id == 91 or menu_id == 92):
            #create an entry for the current config if it doesn't exist
            if not current_config in self.post_test_questions:
                self.post_test_questions[current_config] = {'time': '', 'engaged': '', 'part_of_team': ''}
                #store the order the configs were created (and assumed run in!)
                self.config_run_order.append(current_config)
            
            if menu_id == 91:
                self.post_test_questions[current_config]['time'] = data['time']
            else:
                self.post_test_questions[current_config]['engaged'] = data['engaged']
                self.post_test_questions[current_config]['part_of_team'] = data['part_of_team']

        else:
            #there's no content to save
            pass

        return

    
    def pickleLog(self, file_name):

        #create the directory if it doesn't exist
        import os
        if not os.path.exists(file_name):
            os.makedirs(file_name)

        import pickle
        data = [self.session_id, self.user_details]
        fileo = open(f'{file_name}/user_details.pkl', 'wb')
        pickle.dump(data,fileo)
        fileo.close()

        data = [self.session_id, self.config_run_order, self.post_test_questions]
        fileo = open(f'{file_name}/post_test_responses.pkl', 'wb')
        pickle.dump(data,fileo)
        fileo.close()
        return

        