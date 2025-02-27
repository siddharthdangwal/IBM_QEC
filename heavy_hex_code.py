import numpy as np

class HeavyHexCode:
    '''
    A class to generate one instance of the heavy-hex code
    '''

    def __init__(self, *, code_distance, num_rounds, basis,
                after_clifford_depolarization,
                after_reset_flip_probability,
                before_measure_flip_probability,
                before_round_data_depolarization):
        
        # code parameters
        self.cd=code_distance
        self.nr=num_rounds
        self.basis=basis
        
        # error parameters
        self.acd=after_clifford_depolarization
        self.arfp=after_reset_flip_probability
        self.bmfp=before_measure_flip_probability
        self.brdd=before_round_data_depolarization
        
        # define the qubit-types
        self.data_qubits=None # data qubits
        self.x_gauge_qubits=None # these act as the X stabilizer qubits
        self.flag_qubits=None # these act as both the Z stabilizer qubits and the flag qubits
        self._label_qubits()
        
        # define the CNOT sets
        self.second_cycle_pairs=None
        self.third_cycle_pairs=None
        self.fourth_cycle_pairs=None
        self.fifth_cycle_pairs=None
        
        self.eighth_cycle_pairs=None
        self.ninth_cycle_pairs=None
        self.tenth_cycle_pairs=None
        self._get_cnot_sets(self.x_gauge_qubits, self.data_qubits)
        
        # measurement history
        self.total_measurement_history={i:[] for i in self.data_qubits+self.x_gauge_qubits+self.flag_qubits}
        self.current_measurement_counter=0
    
    # called during initialization
    def _label_qubits(self):
        '''
        For all physical qubits in the code, label each qubit with a unique
        ID. And sort the qubits out according to their functionality - i.e.
        data_qubit/x_gauge_qubit/flag_qubit
        '''
        data_qubits=[]
        x_gauge_qubits=[]
        flag_qubits=[]
        
        code_distance=self.cd    
        n_rows=2*code_distance-1
        n_cols=2*code_distance-1
        
        for i in range(n_rows):
            for j in range(n_cols):
                
                qubit_label=n_cols*i+j
                
                if i%2==0 and j%2==0: # the data qubits
                    data_qubits.append(qubit_label)
                elif ((i%4==1 or i==n_rows-1) and j%4==3) or ((i%4==3 or i==0) and j%4==1): # the X gauge qubits in the bulk
                    x_gauge_qubits.append(qubit_label)
                elif (j%2==0 and i%2==1): # the flag qubits
                    flag_qubits.append(qubit_label)
                else:
                    continue
        
        self.data_qubits=data_qubits
        self.x_gauge_qubits=x_gauge_qubits
        self.flag_qubits=flag_qubits
    
    def _get_cnot_sets(self, x_gauge_qubits, data_qubits):
        '''
        Args:
        data_qubits: The data qubits
        '''
        code_distance=self.cd    
        n_rows=2*code_distance-1
        n_cols=2*code_distance-1
        
        # before applying the X gauge checks, we categorize the qubits into the different sets
        # this convention is according to Fig 2 of Chamberland et al - arxiv 1907.09528v2
        second_cycle_pairs=[]
        third_cycle_pairs=[]
        fourth_cycle_pairs=[]
        fifth_cycle_pairs=[]
        sixth_cycle_pairs=[]
        
        eighth_cycle_pairs=[]
        ninth_cycle_pairs=[]
        tenth_cycle_pairs=[]
        
        for i in range(n_rows):
            for j in range(n_cols):
                if j%2==0 and i%2==1:
                    qubit_label=n_rows*i+j
                    
                    # the bulk x-gauge checks
                    if qubit_label-1 in x_gauge_qubits:
                        second_cycle_pairs.append((qubit_label-1, qubit_label))
                        fifth_cycle_pairs.append((qubit_label-1, qubit_label))
                        if qubit_label-n_cols in data_qubits:
                            third_cycle_pairs.append((qubit_label, qubit_label-n_cols))
                            eighth_cycle_pairs.append((qubit_label-n_cols, qubit_label))
                        if qubit_label+n_cols in data_qubits:
                            fourth_cycle_pairs.append((qubit_label, qubit_label+n_cols))
                            ninth_cycle_pairs.append((qubit_label+n_cols, qubit_label))
                    if qubit_label+1 in x_gauge_qubits:
                        third_cycle_pairs.append((qubit_label+1, qubit_label))
                        sixth_cycle_pairs.append((qubit_label+1, qubit_label))
                        if qubit_label+n_cols in data_qubits:
                            fourth_cycle_pairs.append((qubit_label, qubit_label+n_cols))
                            ninth_cycle_pairs.append((qubit_label+n_cols, qubit_label))
                        if qubit_label-n_cols in data_qubits:
                            fifth_cycle_pairs.append((qubit_label, qubit_label-n_cols))
                            tenth_cycle_pairs.append((qubit_label-n_cols, qubit_label))
                    if not(qubit_label+1 in x_gauge_qubits) and not(qubit_label-1 in x_gauge_qubits):
                        if j==0:
                            if qubit_label-n_cols in data_qubits:
                                eighth_cycle_pairs.append((qubit_label-n_cols, qubit_label))
                            if qubit_label+n_cols in data_qubits:
                                ninth_cycle_pairs.append((qubit_label+n_cols, qubit_label))
                        elif j==n_cols-1:
                            if qubit_label-n_cols in data_qubits:
                                tenth_cycle_pairs.append((qubit_label-n_cols, qubit_label))
                            if qubit_label+n_cols in data_qubits:
                                ninth_cycle_pairs.append((qubit_label+n_cols, qubit_label))
                    
                # bacon-strip checks
                elif i==0 and j%4==2:
                    qubit_label=n_rows*i+j
                    if qubit_label-1 in x_gauge_qubits:
                        fourth_cycle_pairs.append((qubit_label-1, qubit_label))
                elif i==0 and j%4==0:
                    qubit_label=n_rows*i+j
                    if qubit_label+1 in x_gauge_qubits:
                        fifth_cycle_pairs.append((qubit_label+1, qubit_label))
                elif i==n_rows-1 and j%4==2:
                    qubit_label=n_rows*i+j
                    if qubit_label+1 in x_gauge_qubits:
                        sixth_cycle_pairs.append((qubit_label+1, qubit_label))
                elif i==n_rows-1 and j%4==0:
                    qubit_label=n_rows*i+j
                    if qubit_label-1 in x_gauge_qubits:
                        fifth_cycle_pairs.append((qubit_label-1, qubit_label))
        
        # label the CNOT sets
        self.second_cycle_pairs=second_cycle_pairs
        self.third_cycle_pairs=third_cycle_pairs
        self.fourth_cycle_pairs=fourth_cycle_pairs
        self.fifth_cycle_pairs=fifth_cycle_pairs
        self.sixth_cycle_pairs=sixth_cycle_pairs
        
        self.eighth_cycle_pairs=eighth_cycle_pairs
        self.ninth_cycle_pairs=ninth_cycle_pairs
        self.tenth_cycle_pairs=tenth_cycle_pairs
        
   
    def define_qubits(self):
        '''
        Initialize the qubits -- this function works
        '''
        code_distance=self.cd    
        n_cols=2*code_distance-1
        
        codeblock=""""""
        
        for qubit_label in self.data_qubits:
            i=qubit_label//n_cols
            j=qubit_label%n_cols
            codeblock+="""QUBIT_COORDS("""+str(i)+""", """+str(j)+""") """+str(qubit_label)+"""\n"""
        
        for qubit_label in self.x_gauge_qubits:
            i=qubit_label//n_cols
            j=qubit_label%n_cols
            codeblock+="""QUBIT_COORDS("""+str(i)+""", """+str(j)+""") """+str(qubit_label)+"""\n"""
        
        for qubit_label in self.flag_qubits:
            i=qubit_label//n_cols
            j=qubit_label%n_cols
            codeblock+="""QUBIT_COORDS("""+str(i)+""", """+str(j)+""") """+str(qubit_label)+"""\n"""
        
        return codeblock

    def reset_qubits(self, qubits, reset_basis):
        '''
        Reset the qubits - this could be the data qubits, the X gauge qubits or the flag qubits
        '''
        # reset all qubits to 0
        if reset_basis=='Z':
            codeblock="""R"""
        elif reset_basis=='X':
            codeblock="""RX"""
        else:
            raise ValueError("Invalid reset basis")
        
        for el in qubits:
            codeblock+=""" """+str(el)
        codeblock+="""\n"""
        
        return codeblock

    
    def apply_h_gate(self, qubits):
        '''
        Apply the Hadamard gate to the qubits
        '''
        codeblock="""H"""
        for el in qubits:
            codeblock+=""" """+str(el)
        codeblock+="""\n"""
        return codeblock

    def apply_cnots(self, qubit_pairs):
        '''
        Apply CNOT gates to the qubit pairs
        '''
        codeblock="""CNOT"""
        for el in qubit_pairs:
            codeblock+=""" """+str(el[0])+""" """+str(el[1])
        codeblock+="""\n"""
        return codeblock
    
    def apply_mr(self, qubits):
        '''
        Apply the measurement and reset operation to the qubits
        '''
        codeblock="""MR"""
        for el in qubits:
            codeblock+=""" """+str(el)
            self.total_measurement_history[el].append(self.current_measurement_counter)
            self.current_measurement_counter+=1
        codeblock+="""\n"""
        return codeblock

    
    def apply_x_err(self, qubits, p_err):
        '''
        Inserts an X error
        '''
        codeblock="""X_ERROR("""+str(p_err)+""")"""
        for el in qubits:
            codeblock+=""" """+str(el)
        codeblock+="""\n"""
        return codeblock
    
    def apply_z_err(self, qubits, p_err):
        '''
        Inserts a Z error
        '''
        codeblock="""Z_ERROR("""+str(p_err)+""")"""
        for el in qubits:
            codeblock+=""" """+str(el)
        codeblock+="""\n"""
        return codeblock

    
    def apply_flip_error(self, basis, qubits, p_err):
        '''
        Inserts a flip. 
        '''
        if basis=='X':
            codeblock=self.apply_z_err(qubits, p_err)
        elif basis=='Z':
            codeblock=self.apply_x_err(qubits, p_err)
        return codeblock
    
    def apply_one_qb_depolarization_err(self, qubits, p_err):
        '''
        Inserts single qubit depolarizing error
        '''
        codeblock="""DEPOLARIZE1("""+str(p_err)+""")"""
        for el in qubits:
            codeblock+=""" """+str(el)
        codeblock+="""\n"""
        return codeblock

    def apply_two_qb_depolarization_err(self, qubit_pairs, p_err):
        '''
        Inserts two qubit depolarizing error
        '''
        codeblock="""DEPOLARIZE2("""+str(p_err)+""")"""
        for el in qubit_pairs:
            codeblock+=""" """+str(el[0])+""" """+str(el[1])
        codeblock+="""\n"""
        return codeblock

    
    def apply_x_checks(self, *, measure_flags):
        '''
        Args:
        measure_flags: Whether to measure the flag qubits or not
        Generates instruction for the X gauge checks
        '''
        flag_qubits=self.flag_qubits
        x_gauge_qubits=self.x_gauge_qubits
        
        second_cycle_pairs=self.second_cycle_pairs
        third_cycle_pairs=self.third_cycle_pairs
        fourth_cycle_pairs=self.fourth_cycle_pairs
        fifth_cycle_pairs=self.fifth_cycle_pairs
        sixth_cycle_pairs=self.sixth_cycle_pairs
        
        after_clifford_depolarization=self.acd
        after_reset_flip_probability=self.arfp
        before_measure_flip_probability=self.bmfp
        
        codeblock=""""""
        
        # apply the first layer of hadamards
        c1=self.apply_h_gate(x_gauge_qubits)
        codeblock+=c1
        if after_clifford_depolarization>0.0:
            c2=self.apply_one_qb_depolarization_err(x_gauge_qubits, after_clifford_depolarization)
            codeblock+=c2
            
        # apply second cycle operations
        c1=self.apply_cnots(second_cycle_pairs)
        codeblock+=c1
        if after_clifford_depolarization>0.0:
            c2=self.apply_two_qb_depolarization_err(second_cycle_pairs, after_clifford_depolarization)
            codeblock+=c2
        
        # insert tick
        codeblock+="""TICK\n"""
        
        # apply third cycle operations
        c1=self.apply_cnots(third_cycle_pairs)
        codeblock+=c1
        if after_clifford_depolarization>0.0:
            c2=self.apply_two_qb_depolarization_err(third_cycle_pairs, after_clifford_depolarization)
            codeblock+=c2
        
        # insert tick
        codeblock+="""TICK\n"""
        
        # apply fourth cycle operations
        c1=self.apply_cnots(fourth_cycle_pairs)
        codeblock+=c1
        if after_clifford_depolarization>0.0:
            c2=self.apply_two_qb_depolarization_err(fourth_cycle_pairs, after_clifford_depolarization)
            codeblock+=c2
        
        # insert tick
        codeblock+="""TICK\n"""
        
        # apply fifth cycle operations
        c1=self.apply_cnots(fifth_cycle_pairs)
        codeblock+=c1
        if after_clifford_depolarization>0.0:
            c2=self.apply_two_qb_depolarization_err(fifth_cycle_pairs, after_clifford_depolarization)
            codeblock+=c2
        
        # insert tick
        codeblock+="""TICK\n"""
        
        # apply sixth cycle operations
        c1=self.apply_cnots(sixth_cycle_pairs)
        codeblock+=c1
        if after_clifford_depolarization>0.0:
            c2=self.apply_two_qb_depolarization_err(sixth_cycle_pairs, after_clifford_depolarization)
            codeblock+=c2
        
        # insert tick
        codeblock+="""TICK\n"""
        
        # apply hadamard on the x gauge qubit
        c1=self.apply_h_gate(x_gauge_qubits)
        codeblock+=c1
        if after_clifford_depolarization>0.0:
            c2=self.apply_one_qb_depolarization_err(x_gauge_qubits, after_clifford_depolarization)
            codeblock+=c2
        
        # insert tick
        codeblock+="""TICK\n"""
        
        # measure the flag qubits and x gauge qubits
        if before_measure_flip_probability>0.0:
            c1=self.apply_x_err(flag_qubits+x_gauge_qubits, before_measure_flip_probability)
            codeblock+=c1
        
        # final measurement
        if measure_flags:
            c1=self.apply_mr(flag_qubits+x_gauge_qubits)
            codeblock+=c1
            
            # after reset flip
            if after_reset_flip_probability>0.0:
                c1=self.apply_x_err(flag_qubits+x_gauge_qubits, after_reset_flip_probability)
                codeblock+=c1
        else:
            c1=self.apply_mr(x_gauge_qubits)
            codeblock+=c1
            
            # after reset flip
            if after_reset_flip_probability>0.0:
                c1=self.apply_x_err(x_gauge_qubits, after_reset_flip_probability)
                codeblock+=c1
        
        return codeblock

    def apply_z_checks(self):
        '''
        Generates instruction for the Z gauge checks
        '''
        eighth_cycle_pairs=self.eighth_cycle_pairs
        ninth_cycle_pairs=self.ninth_cycle_pairs
        tenth_cycle_pairs=self.tenth_cycle_pairs
        flag_qubits=self.flag_qubits
        
        after_clifford_depolarization=self.acd
        after_reset_flip_probability=self.arfp
        before_measure_flip_probability=self.bmfp
        
        codeblock=""""""
        
        # apply the eighth cycle operations
        c1=self.apply_cnots(eighth_cycle_pairs)
        codeblock+=c1
        if after_clifford_depolarization>0.0:
            c2=self.apply_two_qb_depolarization_err(eighth_cycle_pairs, after_clifford_depolarization)
            codeblock+=c2
        
        # insert tick
        codeblock+="""TICK\n"""
        
        # apply the ninth cycle operations
        c1=self.apply_cnots(ninth_cycle_pairs)
        codeblock+=c1
        if after_clifford_depolarization>0.0:
            c2=self.apply_two_qb_depolarization_err(ninth_cycle_pairs, after_clifford_depolarization)
            codeblock+=c2
        
        # insert tick
        codeblock+="""TICK\n"""
        
        # apply the tenth cycle operations
        c1=self.apply_cnots(tenth_cycle_pairs)
        codeblock+=c1
        if after_clifford_depolarization>0.0:
            c2=self.apply_two_qb_depolarization_err(tenth_cycle_pairs, after_clifford_depolarization)
            codeblock+=c2
        
        # insert tick
        codeblock+="""TICK\n"""
        
        # measure the flag qubits
        if before_measure_flip_probability>0.0:
            c1=self.apply_x_err(flag_qubits, before_measure_flip_probability)
            codeblock+=c1
        
        c1=self.apply_mr(flag_qubits)
        codeblock+=c1
        
        # after reset flip
        if after_reset_flip_probability>0.0:
            c1=self.apply_x_err(flag_qubits, after_reset_flip_probability)
            codeblock+=c1
        
        return codeblock
    
    
    def apply_measurement_detectors(self, *, qubits_to_detect, parity_factor, round_num, measure_flags=False):
        '''
        Applies detectors to the code
        
        Args:
        qubits_to_detect: The indices to qubits we need to measure
        parity_factor: How many instances of the qubit measurement should we consider for the detector
        Eg - If parity_factor=1, we consider the most-recent single measurement DETECTOR rec[-1], if
        parity_factor=2, we consider the most-recent two measurements DETECTOR rec[-1] and rec[-2] and so on
        round_num: Which round is currently being processed
        measure_flags: (Only eligible for the flag qubits): If true then the flag qubits are being measured as flag qubits
        otherwise as the Z stabilizer qubits
        '''
        codeblock=""""""
        n_cols=2*self.cd-1
        
        for el in qubits_to_detect:
            
            i=el//n_cols
            j=el%n_cols
            
            if (parity_factor==1 and qubits_to_detect!=self.flag_qubits) or (parity_factor==1 and measure_flags):
                if measure_flags:
                    assert qubits_to_detect==self.flag_qubits
                
                relative_meas_history=self.total_measurement_history[el][-1]-self.current_measurement_counter
                codeblock+="""DETECTOR("""+str(i)+""", """+str(j)+""", """+str(round_num)+""") rec["""+str(relative_meas_history)+"""]\n"""
            elif parity_factor==1 and qubits_to_detect==self.flag_qubits and measure_flags==False:
                if (j==n_cols-1 and i%4==1) or (j==0 and i%4==3):
                    relative_meas_history=self.total_measurement_history[el][-1]-self.current_measurement_counter
                    codeblock+="""DETECTOR("""+str(i)+""", """+str(j)+""", """+str(round_num)+""") rec["""+str(relative_meas_history)+"""]\n"""
                else:
                    if (el+2 in self.flag_qubits) and not((el+1) in self.x_gauge_qubits):
                        relative_meas_history_1=self.total_measurement_history[el][-1]-self.current_measurement_counter
                        relative_meas_history_2=self.total_measurement_history[el+2][-1]-self.current_measurement_counter
                        codeblock+="""DETECTOR("""+str(i)+""", """+str(j+1)+""", """+str(round_num)+""") rec["""+str(relative_meas_history_1)+"""] rec["""+str(relative_meas_history_2)+"""]\n"""
            elif parity_factor==2 and qubits_to_detect==self.flag_qubits:
                if (j==n_cols-1 and i%4==1) or (j==0 and i%4==3):
                    relative_meas_history_1=self.total_measurement_history[el][-1]-self.current_measurement_counter
                    if len(self.total_measurement_history[el])==2:
                        relative_meas_history_2=self.total_measurement_history[el][-2]-self.current_measurement_counter
                    else:
                        relative_meas_history_2=self.total_measurement_history[el][-3]-self.current_measurement_counter
                    codeblock+="""DETECTOR("""+str(i)+""", """+str(j)+""", """+str(round_num)+""") rec["""+str(relative_meas_history_1)+"""] rec["""+str(relative_meas_history_2)+"""]\n"""
                else:
                    if (el+2 in self.flag_qubits) and not((el+1) in self.x_gauge_qubits):
                        relative_meas_history_1=self.total_measurement_history[el][-1]-self.current_measurement_counter
                        relative_meas_history_2=self.total_measurement_history[el+2][-1]-self.current_measurement_counter
                        
                        if len(self.total_measurement_history[el])==2:
                            relative_meas_history_3=self.total_measurement_history[el][-2]-self.current_measurement_counter
                        else:
                            relative_meas_history_3=self.total_measurement_history[el][-3]-self.current_measurement_counter
                        
                        if len(self.total_measurement_history[el+2])==2:
                            relative_meas_history_4=self.total_measurement_history[el+2][-2]-self.current_measurement_counter
                        else:
                            relative_meas_history_4=self.total_measurement_history[el+2][-3]-self.current_measurement_counter
                        
                        codeblock+="""DETECTOR("""+str(i)+""", """+str(j+1)+""", """+str(round_num)+""") rec["""+str(relative_meas_history_1)+"""] rec["""+str(relative_meas_history_2)+"""] rec["""+str(relative_meas_history_3)+"""] rec["""+str(relative_meas_history_4)+"""]\n"""
                        
                
            else:
                relative_meas_history_1=self.total_measurement_history[el][-1]-self.current_measurement_counter
                relative_meas_history_2=self.total_measurement_history[el][-2]-self.current_measurement_counter
                codeblock+="""DETECTOR("""+str(i)+""", """+str(j)+""", """+str(round_num)+""") rec["""+str(relative_meas_history_1)+"""] rec["""+str(relative_meas_history_2)+"""]\n"""
                
        return codeblock
    
    def apply_data_measurement_detectors(self):
        '''
        After the data qubits are measured, we do a final parity check 
        between the stabilizer qubits of the basis in which the code is 
        initialized (and measured) in and the data qubits surrounding
        the stabilizer qubits
        '''
        if self.basis=='Z':
            codeblock=""""""
            n_cols=2*self.cd-1
            for el in self.flag_qubits:
                i=el//n_cols
                j=el%n_cols
                data_qubits_to_check=[el-n_cols, el+n_cols]
                codeblock+="""DETECTOR("""+str(i)+""", """+str(j)+""", """+str(self.nr)+""")"""
                for dq in data_qubits_to_check:
                    assert len(self.total_measurement_history[dq])==1
                    relative_meas_history=self.total_measurement_history[dq][-1]-self.current_measurement_counter
                    codeblock+=""" rec["""+str(relative_meas_history)+"""]"""
                
                relative_meas_history=self.total_measurement_history[el][-2]-self.current_measurement_counter
                codeblock+=""" rec["""+str(relative_meas_history)+"""]"""
                codeblock+="""\n"""
        
        elif self.basis=='X':
            # for the X basis, the x-gauge qubits are the "stabilizers". Each x-gauge qubit
            # has four data qubits surrounding it. We will do a parity check on these
            # sets of five
            codeblock=""""""
            n_cols=2*self.cd-1
            n_rows=2*self.cd-1
            
            for el in self.x_gauge_qubits:
                i=el//n_cols
                j=el%n_cols
                
                # get the data-qubits to check
                if i==0 or i==n_rows-1:
                    data_qubits_to_check=[el-1, el+1]
                else:
                    data_qubits_to_check=[el-n_cols-1, el-n_cols+1, el+n_cols-1, el+n_cols+1]
                
                codeblock+="""DETECTOR("""+str(i)+""", """+str(j)+""", """+str(self.nr)+""")"""
                for dq in data_qubits_to_check:
                    assert len(self.total_measurement_history[dq])==1
                    relative_meas_history=self.total_measurement_history[dq][-1]-self.current_measurement_counter
                    codeblock+=""" rec["""+str(relative_meas_history)+"""]"""
                
                relative_meas_history=self.total_measurement_history[el][-1]-self.current_measurement_counter
                codeblock+=""" rec["""+str(relative_meas_history)+"""]"""
                codeblock+="""\n"""
        
        return codeblock


    def apply_observable_label(self):
        '''
        Gives the observable
        
        According to Fig 2 of Chamberland et al - arxiv 1907.09528v2
        The X logical operator is the vertical one, and the Z operator
        is the horizontal one
        '''
        n_cols=2*self.cd-1
        
        codeblock="""OBSERVABLE_INCLUDE(0)"""
        if self.basis=='X':
            candidate_qubits=[i for i in self.data_qubits if i%n_cols==0] # first column -- logical X observable
            for el in candidate_qubits:
                relative_measurement_history=self.total_measurement_history[el][-1]-self.current_measurement_counter
                codeblock+=""" rec["""+str(relative_measurement_history)+"""]"""
        elif self.basis=='Z':
            candidate_qubits=[i for i in self.data_qubits if i//n_cols==0] # first row -- logical Z observable
            for el in candidate_qubits:
                relative_measurement_history=self.total_measurement_history[el][-1]-self.current_measurement_counter
                codeblock+=""" rec["""+str(relative_measurement_history)+"""]"""
        
        return codeblock
    
    def create_heavy_hex_code(self):
        '''
        Args:
        code_distance: the distance of the heavy-hex code
        rounds: the number of rounds to run the code
        after_clifford_depolarization: the probability of applying a depolarizing error after the Clifford gates
        after_reset_flip_probability: the probability of flipping the qubit after the reset
        before_measure_flip_probability: the probability of flipping the qubit before the measurement
        before_round_data_depolarization: the probability of applying a depolarizing error before the round
        '''
        # number of rows and columns in the code-block
        code_distance=self.cd
        n_rows=2*code_distance-1
        n_cols=2*code_distance-1
        
        full_codeblock=""""""
        
        # define the qubits -- this function looks good
        codeblock=self.define_qubits()
        full_codeblock+=codeblock    
        
        # reset the data qubits
        codeblock=self.reset_qubits(self.data_qubits, reset_basis=self.basis)
        full_codeblock+=codeblock
        if self.arfp>0.0:
            # apply a flip after the reset
            codeblock=self.apply_flip_error(self.basis, self.data_qubits, self.arfp)
            full_codeblock+=codeblock
        
        # reset the X gauge qubits -- the X gauge qubits are always reset in the Z basis
        codeblock=self.reset_qubits(self.x_gauge_qubits, reset_basis='Z')
        full_codeblock+=codeblock
        if self.arfp>0.0:
            # apply a flip after the reset
            codeblock=self.apply_flip_error('Z', self.x_gauge_qubits, self.arfp)
            full_codeblock+=codeblock
        
        # reset the flag qubits - the flag qubits are always reset in the Z basis
        codeblock=self.reset_qubits(self.flag_qubits, reset_basis='Z')
        full_codeblock+=codeblock
        if self.arfp>0.0:
            # apply a flip after the reset
            codeblock=self.apply_flip_error('Z', self.flag_qubits, self.arfp)
            full_codeblock+=codeblock
        
        # insert tick
        full_codeblock+="""TICK\n"""
        
        if self.brdd>0.0:
            # apply depolarizing error before the round
            codeblock=self.apply_one_qb_depolarization_err(self.data_qubits, self.brdd)
            full_codeblock+=codeblock
        
        # initialize the qubits
        if self.basis=='Z': # already in the Z basis, project in X basis as well
            codeblock=self.apply_z_checks()
            full_codeblock+=codeblock
            
            codeblock=self.apply_measurement_detectors(qubits_to_detect=self.flag_qubits, 
                                        parity_factor=1,
                                        round_num=0)
            full_codeblock+=codeblock
            
            codeblock=self.apply_x_checks(measure_flags=False)
            full_codeblock+=codeblock
            
            # codeblock=self.apply_measurement_detectors(qubits_to_detect=self.flag_qubits, 
            #                             parity_factor=1,
            #                             round_num=0)
            # full_codeblock+=codeblock
            
        elif self.basis=='X': # already in the X basis, project in Z basis as well
            codeblock=self.apply_x_checks(measure_flags=False)
            full_codeblock+=codeblock
            
            codeblock=self.apply_measurement_detectors(qubits_to_detect=self.x_gauge_qubits, 
                                        parity_factor=1,
                                        round_num=0)
            full_codeblock+=codeblock
            
            # codeblock=self.apply_measurement_detectors(qubits_to_detect=self.flag_qubits, 
            #                             parity_factor=1,
            #                             round_num=0)
            
            full_codeblock+=codeblock
            
            codeblock=self.apply_z_checks()
            full_codeblock+=codeblock
        else:
            raise ValueError("Invalid basis")
        
        # ----------------------------------- start the first round -----------------------------------
        
        # # apply before-round data depolarization
        # if self.brdd>0.0:
        #     # apply depolarizing error before the round
        #     codeblock=self.apply_one_qb_depolarization_err(self.data_qubits, self.brdd)
        #     full_codeblock+=codeblock
        
        if self.basis=='Z':
            # The first round of Z basis check -- deterministic
            codeblock=self.apply_z_checks()
            full_codeblock+=codeblock
            
            codeblock=self.apply_measurement_detectors(qubits_to_detect=self.flag_qubits, 
                                        parity_factor=2,
                                        round_num=0)
            full_codeblock+=codeblock
            
            # Compare parity with last round of X checks
            codeblock=self.apply_x_checks(measure_flags=True)
            full_codeblock+=codeblock
            
            codeblock=self.apply_measurement_detectors(qubits_to_detect=self.x_gauge_qubits, 
                                    parity_factor=2,
                                    round_num=0)
            full_codeblock+=codeblock
            
            # flag qubit measurements -- always deterministic
            codeblock=self.apply_measurement_detectors(qubits_to_detect=self.flag_qubits, 
                                        parity_factor=1,
                                        round_num=0)
            full_codeblock+=codeblock   
        elif self.basis=='X':
            codeblock=self.apply_x_checks()
            full_codeblock+=codeblock
            
            codeblock=self.apply_measurement_detectors(qubits_to_detect=self.x_gauge_qubits, 
                                        parity_factor=2,
                                        round_num=0)
            full_codeblock+=codeblock
            
            codeblock=self.apply_measurement_detectors(qubits_to_detect=self.flag_qubits, 
                                    parity_factor=1,
                                    round_num=0)
            full_codeblock+=codeblock
            
            codeblock=self.apply_z_checks()
            full_codeblock+=codeblock
            
            codeblock=self.apply_measurement_detectors(qubits_to_detect=self.flag_qubits, 
                                        parity_factor=2,
                                        round_num=0)
            full_codeblock+=codeblock   
        else:
            raise ValueError("Invalid basis")
        
        ######################################### Repeat the block ##############################################
        if self.nr>1:
            print('Inside the repeat block')
            full_codeblock+= "REPEAT "+str(self.nr-1)+""" {\n"""
            temp_codeblock= """\tTICK\n"""
            
            # apply before-round data depolarization
            if self.brdd>0.0:
                # apply depolarizing error before the round
                codeblock=self.apply_one_qb_depolarization_err(self.data_qubits, self.brdd)
                temp_codeblock+=codeblock
            
            if self.basis=='Z':
                # The first round of Z basis check -- deterministic
                codeblock=self.apply_z_checks()
                temp_codeblock+=codeblock
                
                codeblock=self.apply_measurement_detectors(qubits_to_detect=self.flag_qubits, 
                                            parity_factor=2,
                                            round_num=0)
                temp_codeblock+=codeblock
                
                # Compare parity with last round of X checks
                codeblock=self.apply_x_checks(measure_flags=True)
                temp_codeblock+=codeblock
                
                codeblock=self.apply_measurement_detectors(qubits_to_detect=self.x_gauge_qubits, 
                                        parity_factor=2,
                                        round_num=0)
                temp_codeblock+=codeblock
                
                # flag qubit measurements -- always deterministic
                codeblock=self.apply_measurement_detectors(qubits_to_detect=self.flag_qubits, 
                                            parity_factor=1,
                                            round_num=0)
                temp_codeblock+=codeblock   
            elif self.basis=='X':
                codeblock=self.apply_x_checks()
                temp_codeblock+=codeblock
                
                codeblock=self.apply_measurement_detectors(qubits_to_detect=self.x_gauge_qubits, 
                                            parity_factor=2,
                                            round_num=0)
                temp_codeblock+=codeblock
                
                codeblock=self.apply_measurement_detectors(qubits_to_detect=self.flag_qubits, 
                                        parity_factor=1,
                                        round_num=0)
                temp_codeblock+=codeblock
                
                codeblock=self.apply_z_checks()
                temp_codeblock+=codeblock
                
                codeblock=self.apply_measurement_detectors(qubits_to_detect=self.flag_qubits, 
                                            parity_factor=2,
                                            round_num=0)
                temp_codeblock+=codeblock   
            else:
                raise ValueError("Invalid basis")
        
            # insert the tab
            temp_codeblock=temp_codeblock.replace("\n", "\n\t")
            temp_codeblock+="""}\n"""
            full_codeblock+=temp_codeblock
        
        # measure the data qubits
        codeblock=self.apply_flip_error(basis=self.basis, qubits=self.data_qubits, p_err=self.bmfp)
        full_codeblock+=codeblock
        
        if self.basis=='X':
            codeblock="""MX"""
            for el in self.data_qubits:
                codeblock+=""" """+str(el)
                self.total_measurement_history[el].append(self.current_measurement_counter)
                self.current_measurement_counter+=1
        elif self.basis=='Z':
            codeblock="""M"""
            for el in self.data_qubits:
                codeblock+=""" """+str(el)
                self.total_measurement_history[el].append(self.current_measurement_counter)
                self.current_measurement_counter+=1
        else:
            raise ValueError("Invalid basis")
        
        codeblock+="""\n"""
        full_codeblock+=codeblock
        
        # get the data-measurement detectors
        codeblock=self.apply_data_measurement_detectors()
        full_codeblock+=codeblock
        
        # get the observable
        codeblock=self.apply_observable_label()
        full_codeblock+=codeblock
        
        return full_codeblock

    def test_function(self):
        
        code_distance=self.cd
        n_rows=2*code_distance-1
        n_cols=2*code_distance-1
        
        full_codeblock=""""""
        
        # define the qubits -- this function looks good
        codeblock=self.define_qubits()
        full_codeblock+=codeblock    
        
        # reset the data qubits
        codeblock=self.reset_qubits(self.data_qubits, reset_basis=self.basis)
        full_codeblock+=codeblock
        if self.arfp>0.0:
            # apply a flip after the reset
            codeblock=self.apply_flip_error(self.basis, self.data_qubits, self.arfp)
            full_codeblock+=codeblock
        
        # reset the X gauge qubits -- the X gauge qubits are always reset in the Z basis
        codeblock=self.reset_qubits(self.x_gauge_qubits, reset_basis='Z')
        full_codeblock+=codeblock
        if self.arfp>0.0:
            # apply a flip after the reset
            codeblock=self.apply_flip_error('Z', self.x_gauge_qubits, self.arfp)
            full_codeblock+=codeblock
        
        # reset the flag qubits - the flag qubits are always reset in the Z basis
        codeblock=self.reset_qubits(self.flag_qubits, reset_basis='Z')
        full_codeblock+=codeblock
        if self.arfp>0.0:
            # apply a flip after the reset
            codeblock=self.apply_flip_error('Z', self.flag_qubits, self.arfp)
            full_codeblock+=codeblock
        
        # insert tick
        full_codeblock+="""TICK\n"""
        
        if self.brdd>0.0:
            # apply depolarizing error before the round
            codeblock=self.apply_one_qb_depolarization_err(self.data_qubits, self.brdd)
            full_codeblock+=codeblock
        
        # add the parity checks
        # codeblock=self.apply_z_checks()
        # full_codeblock+=codeblock
        
        # codeblock=self.apply_measurement_detectors(qubits_to_detect=self.flag_qubits, 
        #                             parity_factor=1,
        #                             round_num=0)
        # full_codeblock+=codeblock
        
        codeblock=self.apply_x_checks(measure_flags=False)
        full_codeblock+=codeblock
        
        codeblock=self.apply_z_checks()
        full_codeblock+=codeblock
        
        codeblock=self.apply_measurement_detectors(qubits_to_detect=self.flag_qubits, 
                                    parity_factor=1,
                                    round_num=0)
        full_codeblock+=codeblock
        
        # Compare parity with last round of X checks
        codeblock=self.apply_x_checks(measure_flags=True)
        full_codeblock+=codeblock
        
        codeblock=self.apply_measurement_detectors(qubits_to_detect=self.x_gauge_qubits, 
                                parity_factor=2,
                                round_num=0)
        full_codeblock+=codeblock
        
        # flag qubit measurements -- always deterministic
        # codeblock=self.apply_measurement_detectors(qubits_to_detect=self.flag_qubits, 
        #                             parity_factor=1,
        #                             round_num=0, measure_flags=True)
        # full_codeblock+=codeblock
        
        return full_codeblock
        

############################################### ALL THE DECODERS ############################################################################

# the basic decoder check
def process_syndrome(syndrome, observable):
    '''
    Args:
    syndrome: the syndrome measurement results - This will be a list of 
    True and False covering all the rounds for one shot. Our task is to 
    first block the syndromes into different rounds and then create a 
    "final syndrom value" corresponding to each measurement qubit
    '''
    return

def weight_one_error_clique(syndromes, observable_flips):
    '''
    Args:
    syndromes: The syndromes extracted from the measurement outcomes
    
    This function assumes that we have a final single round of syndrome info.
    We will use the Clique decoder to find only weight-one errors and correct 
    them
    ''' 
    return

def lut_decoder(syndromes, observable_flips):
    '''
    '''
    return


# create an instance of the heavy-hex code
# hhc=HeavyHexCode(
#     code_distance=3,
#     num_rounds=1,
#     basis='Z',
#     after_clifford_depolarization=0,
#     after_reset_flip_probability=0,
#     before_measure_flip_probability=0,
#     before_round_data_depolarization=0,
# )

# codeblock=hhc.create_heavy_hex_code()