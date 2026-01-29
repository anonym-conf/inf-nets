import helpers, optimizers, loader, evaluator, calibrators, cost_calculator, copy, pprint, json
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser()
parser.add_argument('--sLLM', default='gemma3-1b-it', type=str,
                    choices=['gemma3-1b-it', 'gemma3-4b-it', 'gemma3-12b-it', 'qwen3-4b-it', 'ministral3-3b-it'], help='Name of sLLM.')
parser.add_argument('--sLLM_param_count', type=int, default=1, help='Number of parameters of sLLM (in billions)')
parser.add_argument('--mLLM', default='gemma3-4b-it', type=str,
                    choices=['gemma3-4b-it', 'gemma3-12b-it', 'gemma3-27b-it', 'qwen3-4b-it', 'ministral3-3b-it'], help='Name of mLLM.')
parser.add_argument('--mLLM_param_count', type=int, default=4, help='Number of parameters of mLLM (in billions)')
parser.add_argument('--dataset', default='sst2', type=str,
                    choices=['sst2', 'emotion', 'agnews', 'fakenews', 'squad'], help='Name of dataset.')
parser.add_argument('--calibrate', default=False, action='store_true',
                    help='Calibrate probabilities.')
parser.add_argument('--xi', default=0.95, type=float, help='Accuracy goal.')
parser.add_argument('--single_threshold', default=False, action='store_true', help='Kind of policy.')
parser.add_argument('--oracle', default=False, action='store_true', help='Oracle setting?')
parser.add_argument('--calculate_cost', default=False, action='store_true', help='Calculate cost post-hoc?')
parser.add_argument('--save_results', default=False, action='store_true', help='Save results?')

def run_multi_thresholds(args_):
    train_loader = loader.StructuresLoader(
        dataset_name=args_.dataset, s_llm=args_.sLLM, m_llm=args_.mLLM
    )
    test_loader = loader.StructuresLoader(
        dataset_name=args_.dataset, s_llm=args_.sLLM, m_llm=args_.mLLM
    )

    train_loader.load_dataset()  # loads dataset

    train_loader.sLLM_results = train_loader.load_results_file(f'.../{args_.dataset}_train_{args_.sLLM}_profiler.pkl')
    train_loader.mLLM_results = train_loader.load_results_file(f'.../{args_.dataset}_train_{args_.mLLM}_profiler.pkl')
    print(f'\nLoaded initial file. sLLM: {len(train_loader.sLLM_results)} samples.')
    print(f'Loaded initial file. mLLM: {len(train_loader.mLLM_results)} samples.\n')

    split_idx = int(len(train_loader.sLLM_results) * 0.8)
    test_portion_s = train_loader.sLLM_results[split_idx:]
    test_portion_m = train_loader.mLLM_results[split_idx:]
    # first 80% → training

    train_loader.sLLM_results = copy.deepcopy(train_loader.sLLM_results[:split_idx])
    train_loader.mLLM_results = copy.deepcopy(train_loader.mLLM_results[:split_idx])
    
    # last 20% → testing
    test_loader.sLLM_results = copy.deepcopy(test_portion_s)
    test_loader.mLLM_results = copy.deepcopy(test_portion_m)
    del test_portion_s, test_portion_m

    print(f'After spliting --> sLLM: {len(train_loader.sLLM_results)} training samples / {len(test_loader.sLLM_results)} testing samples.')
    print(f'After spliting --> mLLM: {len(train_loader.mLLM_results)} training samples / {len(test_loader.mLLM_results)} testing samples.')
    assert len(train_loader.sLLM_results) == len(train_loader.mLLM_results)
    assert len(test_loader.sLLM_results) == len(test_loader.mLLM_results)

    if args_.calculate_cost:
        cost_calc = cost_calculator.CostCalculator(results=train_loader.sLLM_results, model_id=helpers.get_cost_model(args_.sLLM))
        cost_calc.calculate()
        train_loader.sLLM_results = copy.deepcopy(cost_calc.results)
        del cost_calc

        cost_calc = cost_calculator.CostCalculator(results=train_loader.mLLM_results, model_id=helpers.get_cost_model(args_.mLLM))
        cost_calc.calculate()
        train_loader.mLLM_results = copy.deepcopy(cost_calc.results)
        del cost_calc

        sLLM_cost = helpers.cost_distribution_signature(train_loader.get_result_values(train_loader.sLLM_results,
                                                                                       key='prof_post_hoc_cost'))
        mLLM_cost = helpers.cost_distribution_signature(train_loader.get_result_values(train_loader.mLLM_results,
                                                                                       key='prof_post_hoc_cost'))
        print(f'\nCost signatures: sLLM = {sLLM_cost}, mLLM = {mLLM_cost}.\n')
        sLLM_cost_param_aware = helpers.cost_distribution_signature_param_aware(
            train_loader.get_result_values(train_loader.sLLM_results, key='prof_post_hoc_cost'), parameter_count=args_.sLLM_param_count)
        mLLM_cost_param_aware = helpers.cost_distribution_signature_param_aware(
            train_loader.get_result_values(train_loader.mLLM_results, key='prof_post_hoc_cost'), parameter_count=args_.mLLM_param_count)
        print(f'Cost signatures (param-aware): sLLM = {sLLM_cost_param_aware}, mLLM = {mLLM_cost_param_aware}.\n')
    else:
        sLLM_cost = helpers.cost_distribution_signature(train_loader.get_result_values(train_loader.sLLM_results,
                                                                                       key='cost_tflops'))
        mLLM_cost = helpers.cost_distribution_signature(train_loader.get_result_values(train_loader.mLLM_results,
                                                                                       key='cost_tflops'))
        print(f'\nCost signatures: sLLM = {sLLM_cost}, mLLM = {mLLM_cost}.\n')
        sLLM_cost_param_aware = helpers.cost_distribution_signature_param_aware(
            train_loader.get_result_values(train_loader.sLLM_results, key='cost_tflops'), parameter_count=args_.sLLM_param_count)
        mLLM_cost_param_aware = helpers.cost_distribution_signature_param_aware(
            train_loader.get_result_values(train_loader.mLLM_results, key='cost_tflops'), parameter_count=args_.mLLM_param_count)
        print(f'Cost signatures (param-aware): sLLM = {sLLM_cost_param_aware}, mLLM = {mLLM_cost_param_aware}.\n')

    
    gold_truth_train = train_loader.get_result_values(train_loader.sLLM_results, key='ground_truth_label')
    assert gold_truth_train == train_loader.get_result_values(train_loader.mLLM_results, key='ground_truth_label')
    # print(gold_truth_train == gold_truth)

    sLLM_pred_train = train_loader.get_result_values(train_loader.sLLM_results, key='pred_idx')

    mLLM_pred_train = train_loader.get_result_values(train_loader.mLLM_results, key='pred_idx')

    # eval_ = evaluator.Evaluator()
    evaluator.Evaluator().classification_metrics(gold_truth_train, sLLM_pred_train, model_name=args_.sLLM)
    acc_bound = evaluator.Evaluator().classification_metrics(gold_truth_train, mLLM_pred_train, 
                                                             model_name=args_.mLLM)
    evaluator.Evaluator().classification_metrics(mLLM_pred_train, sLLM_pred_train, 
                                                 model_name=args_.sLLM, oracle=args_.mLLM)
    # del eval_

    args_.xi = acc_bound   # override bound
    print(f'New bound: {args_.xi}, target error: {1. - args_.xi}')

    calibrator = None
    if args_.calibrate:
        if args_.dataset not in ['sst2', 'fakenews']:
            probs = train_loader.obtain_multiclass_betas(return_all=True)
        else:
            probs = train_loader.obtain_binary_betas(return_both=True)
            
        # calibrator = calibrators.TopConfidenceCalibrator()
        calibrator = calibrators.PerClassConfidenceCalibrator(probs.shape[1])
        print(f'\nCalibrating confidence...')
        calibrator.fit(probs, gold_truth_train, y_hat=sLLM_pred_train)
        _, sLLM_betas_train = calibrator.predict(probs)
    else:
        # sLLM_betas_train = train_loader.obtain_binary_betas()
        if args_.dataset not in ['sst2', 'fakenews']:
            sLLM_betas_train = train_loader.obtain_multiclass_betas()
        else:
            sLLM_betas_train = train_loader.obtain_binary_betas()

    

    optimizer = optimizers.MultiThresholdOptimizer(y_true=gold_truth_train, 
                                                   y_hat_s=sLLM_pred_train, y_hat_m=mLLM_pred_train,
                                                   betas=sLLM_betas_train, xi=args_.xi, oracle=args_.oracle,
                                                   name='oracle' if args_.oracle else 'non_oracle')

    
    opt_result = optimizer.optimize(c_s=sLLM_cost_param_aware, c_m=mLLM_cost_param_aware)
   
    tau_star, _ = opt_result

    eval_train = evaluator.Evaluator(
        y_true=gold_truth_train, y_hat_s=sLLM_pred_train, y_hat_m=mLLM_pred_train,
        betas=sLLM_betas_train, threshold=tau_star, xi=args_.xi, oracle=args_.oracle
    )

    if args_.oracle:
        eval_train.coverage_metrics(c_s=sLLM_cost_param_aware, c_m=mLLM_cost_param_aware)
        # eval_train.threshold_plots(c_s=sLLM_cost_param_aware, c_m=mLLM_cost_param_aware,
        #                            error_ind=err_star, cost_ind=cost_star)

    eval_train.coverage_metrics_non_oracle(c_s=sLLM_cost_param_aware, c_m=mLLM_cost_param_aware, 
                                           gold_truth=gold_truth_train)
    print(f'\nStatistics for training set:')
    pprint.pprint(eval_train.evaluate_policy(
        c_s=sLLM_cost_param_aware, c_m=mLLM_cost_param_aware,
        mode='oracle' if args_.oracle else 'non_oracle'
    ))
    print()
    


    if args_.calculate_cost:
        cost_calc = cost_calculator.CostCalculator(results=test_loader.sLLM_results, model_id=helpers.get_cost_model(args_.sLLM))
        cost_calc.calculate()
        train_loader.sLLM_results = copy.deepcopy(cost_calc.results)
        del cost_calc

        cost_calc = cost_calculator.CostCalculator(results=test_loader.mLLM_results, model_id=helpers.get_cost_model(args_.mLLM))
        cost_calc.calculate()
        train_loader.mLLM_results = copy.deepcopy(cost_calc.results)
        del cost_calc

        sLLM_cost_test = helpers.cost_distribution_signature(test_loader.get_result_values(test_loader.sLLM_results,
                                                                                       key='prof_post_hoc_cost'))
        mLLM_cost_test = helpers.cost_distribution_signature(test_loader.get_result_values(test_loader.mLLM_results,
                                                                                           key='prof_post_hoc_cost'))
        print(f'\nCost signatures - test: sLLM = {sLLM_cost_test}, mLLM = {mLLM_cost_test}.\n')
        sLLM_cost_param_aware_test = helpers.cost_distribution_signature_param_aware(
            test_loader.get_result_values(test_loader.sLLM_results, key='prof_post_hoc_cost'), 
            parameter_count=args_.sLLM_param_count)
        mLLM_cost_param_aware_test = helpers.cost_distribution_signature_param_aware(
            test_loader.get_result_values(test_loader.mLLM_results, key='prof_post_hoc_cost'), 
            parameter_count=args_.mLLM_param_count)
        print(f'Cost signatures (param-aware) - test: sLLM = {sLLM_cost_param_aware_test},'
              f' mLLM = {mLLM_cost_param_aware_test}.\n')
    else:
        sLLM_cost_test = helpers.cost_distribution_signature(test_loader.get_result_values(test_loader.sLLM_results,
                                                                                           key='cost_tflops'))
        mLLM_cost_test = helpers.cost_distribution_signature(test_loader.get_result_values(test_loader.mLLM_results,
                                                                                           key='cost_tflops'))
        print(f'\nCost signatures - test: sLLM = {sLLM_cost_test}, mLLM = {mLLM_cost_test}.\n')
        sLLM_cost_param_aware_test = helpers.cost_distribution_signature_param_aware(
            test_loader.get_result_values(test_loader.sLLM_results, key='cost_tflops'), parameter_count=1)
        mLLM_cost_param_aware_test = helpers.cost_distribution_signature_param_aware(
            test_loader.get_result_values(test_loader.mLLM_results, key='cost_tflops'), parameter_count=4)
        print(f'Cost signatures (param-aware) - test: sLLM = {sLLM_cost_param_aware_test},'
              f' mLLM = {mLLM_cost_param_aware_test}.\n')
    
    

    sLLM_pred_test = test_loader.get_result_values(test_loader.sLLM_results, key='pred_idx')
    mLLM_pred_test = test_loader.get_result_values(test_loader.mLLM_results, key='pred_idx')
    if args_.calibrate:
        if args_.dataset not in ['sst2', 'fakenews']:
            probs_test = test_loader.obtain_multiclass_betas(return_all=True)
        else:
            probs_test = test_loader.obtain_binary_betas(return_both=True)
        _, sLLM_betas_test = calibrator.predict(probs_test, y_hat=sLLM_pred_test)
    else:
        if args_.dataset not in ['sst2', 'fakenews']:
            sLLM_betas_test = test_loader.obtain_multiclass_betas()  # returns max...
        else:
            sLLM_betas_test = test_loader.obtain_binary_betas()


    # for non-oracle evaluation...
    
    gold_truth_test = test_loader.get_result_values(test_loader.sLLM_results, key='ground_truth_label')
    
    # eval_ = evaluator.Evaluator()
    evaluator.Evaluator().classification_metrics(gold_truth_test, sLLM_pred_test, model_name=args_.sLLM)
    evaluator.Evaluator().classification_metrics(gold_truth_test, mLLM_pred_test, model_name=args_.mLLM)
    evaluator.Evaluator().classification_metrics(mLLM_pred_test, sLLM_pred_test, 
                                                 model_name=args_.sLLM, oracle=args_.mLLM)
    # del eval_

    eval_test = evaluator.Evaluator(
        y_true=gold_truth_test, y_hat_s=sLLM_pred_test, y_hat_m=mLLM_pred_test,
        betas=sLLM_betas_test, threshold=tau_star, xi=args_.xi, test=True, oracle=args_.oracle
    )

    if args_.oracle:
        err_star_test, cost_star_test = eval_test.coverage_metrics(c_s=sLLM_cost_param_aware_test,
                                                                   c_m=mLLM_cost_param_aware_test)
        # eval_test.threshold_plots(c_s=sLLM_cost_param_aware_test, c_m=mLLM_cost_param_aware_test,
        #                            error_ind=err_star_test, cost_ind=cost_star_test)

    err_star_test_non_oracle, cost_star_test_non_oracle = eval_test.coverage_metrics_non_oracle(
        c_s=sLLM_cost_param_aware_test, c_m=mLLM_cost_param_aware_test, gold_truth=gold_truth_test
    )
    print(f'\nStatistics for testing set:')
    test_eval = eval_test.evaluate_policy(
        c_s=sLLM_cost_param_aware_test, c_m=mLLM_cost_param_aware_test,
        mode='oracle' if args_.oracle else 'non_oracle',
        gold_truth=gold_truth_test if args_.oracle else None
    )
    pprint.pprint(test_eval)
    
    print()
    eval_test.threshold_plots_non_oracle(c_s=sLLM_cost_param_aware_test, c_m=mLLM_cost_param_aware_test,
                                         error_ind=err_star_test_non_oracle, cost_ind=cost_star_test_non_oracle,
                                         gold_truth=gold_truth_test, 
                                         filename_error=f"figures/{args.dataset}_{args.sLLM}_{args.mLLM}_xi{args.xi}_{"oracle" if args.oracle else "non_oracle"}_error_plot.pdf",
                                         filename_cost=f"figures/{args.dataset}_{args.sLLM}_{args.mLLM}_xi{args.xi}_{"oracle" if args.oracle else "non_oracle"}_cost_plot.pdf")

    return test_eval


def run_single_threshold(args_):
    train_loader = loader.StructuresLoader(
        dataset_name=args_.dataset, s_llm=args_.sLLM, m_llm=args_.mLLM
    )
    test_loader = loader.StructuresLoader(
        dataset_name=args_.dataset, s_llm=args_.sLLM, m_llm=args_.mLLM
    )

    train_loader.load_dataset()  # loads dataset

    train_loader.sLLM_results = train_loader.load_results_file(f'.../{args_.dataset}_train_{args_.sLLM}_profiler.pkl')
    train_loader.mLLM_results = train_loader.load_results_file(f'.../{args_.dataset}_train_{args_.mLLM}_profiler.pkl')
    print(f'\nLoaded initial file. sLLM: {len(train_loader.sLLM_results)} samples.')
    print(f'Loaded initial file. mLLM: {len(train_loader.mLLM_results)} samples.\n')

    split_idx = int(len(train_loader.sLLM_results) * 0.8)
    test_portion_s = train_loader.sLLM_results[split_idx:]
    test_portion_m = train_loader.mLLM_results[split_idx:]
    # first 80% → training

    train_loader.sLLM_results = copy.deepcopy(train_loader.sLLM_results[:split_idx])
    train_loader.mLLM_results = copy.deepcopy(train_loader.mLLM_results[:split_idx])
    
    # last 20% → testing
    test_loader.sLLM_results = copy.deepcopy(test_portion_s)
    test_loader.mLLM_results = copy.deepcopy(test_portion_m)
    del test_portion_s, test_portion_m

    print(f'After spliting --> sLLM: {len(train_loader.sLLM_results)} training samples / {len(test_loader.sLLM_results)} testing samples.')
    print(f'After spliting --> mLLM: {len(train_loader.mLLM_results)} training samples / {len(test_loader.mLLM_results)} testing samples.')
    assert len(train_loader.sLLM_results) == len(train_loader.mLLM_results)
    assert len(test_loader.sLLM_results) == len(test_loader.mLLM_results)

    if args_.calculate_cost:
        cost_calc = cost_calculator.CostCalculator(results=train_loader.sLLM_results, model_id=helpers.get_cost_model(args_.sLLM))
        cost_calc.calculate()
        train_loader.sLLM_results = copy.deepcopy(cost_calc.results)
        del cost_calc

        cost_calc = cost_calculator.CostCalculator(results=train_loader.mLLM_results, model_id=helpers.get_cost_model(args_.mLLM))
        cost_calc.calculate()
        train_loader.mLLM_results = copy.deepcopy(cost_calc.results)
        del cost_calc

        sLLM_cost = helpers.cost_distribution_signature(train_loader.get_result_values(train_loader.sLLM_results,
                                                                                       key='prof_post_hoc_cost'))
        mLLM_cost = helpers.cost_distribution_signature(train_loader.get_result_values(train_loader.mLLM_results,
                                                                                       key='prof_post_hoc_cost'))
        print(f'\nCost signatures: sLLM = {sLLM_cost}, mLLM = {mLLM_cost}.\n')
        sLLM_cost_param_aware = helpers.cost_distribution_signature_param_aware(
            train_loader.get_result_values(train_loader.sLLM_results, key='prof_post_hoc_cost'), parameter_count=args_.sLLM_param_count)
        mLLM_cost_param_aware = helpers.cost_distribution_signature_param_aware(
            train_loader.get_result_values(train_loader.mLLM_results, key='prof_post_hoc_cost'), parameter_count=args_.mLLM_param_count)
        print(f'Cost signatures (param-aware): sLLM = {sLLM_cost_param_aware}, mLLM = {mLLM_cost_param_aware}.\n')
    else:
        sLLM_cost = helpers.cost_distribution_signature(train_loader.get_result_values(train_loader.sLLM_results,
                                                                                       key='cost_tflops'))
        mLLM_cost = helpers.cost_distribution_signature(train_loader.get_result_values(train_loader.mLLM_results,
                                                                                       key='cost_tflops'))
        print(f'\nCost signatures: sLLM = {sLLM_cost}, mLLM = {mLLM_cost}.\n')
        sLLM_cost_param_aware = helpers.cost_distribution_signature_param_aware(
            train_loader.get_result_values(train_loader.sLLM_results, key='cost_tflops'), parameter_count=args_.sLLM_param_count)
        mLLM_cost_param_aware = helpers.cost_distribution_signature_param_aware(
            train_loader.get_result_values(train_loader.mLLM_results, key='cost_tflops'), parameter_count=args_.mLLM_param_count)
        print(f'Cost signatures (param-aware): sLLM = {sLLM_cost_param_aware}, mLLM = {mLLM_cost_param_aware}.\n')
    
    gold_truth_train = train_loader.get_result_values(train_loader.sLLM_results, key='gold_answer')
    assert gold_truth_train == train_loader.get_result_values(train_loader.mLLM_results, key='gold_answer')

    sLLM_pred_train = train_loader.get_result_values(train_loader.sLLM_results, key='pred')
    mLLM_pred_train = train_loader.get_result_values(train_loader.mLLM_results, key='pred')

    # eval_ = evaluator.Evaluator()
    evaluator.Evaluator().classification_metrics([1] * len(gold_truth_train), helpers.get_binary_labels(
                                                                y_true=gold_truth_train, y_hat=sLLM_pred_train,
                                                             ), model_name=args_.sLLM)
                                                #  train_loader.get_result_values(train_loader.sLLM_results, key='binary_with_gold'),
                                                #  helpers.get_binary_labels(
                                                #      sim=train_loader.get_result_values(train_loader.sLLM_results, key='sim_with_gold')
                                                #  ), 
                                                #  model_name=args_.sLLM)
    acc_bound = evaluator.Evaluator().classification_metrics([1] * len(gold_truth_train), helpers.get_binary_labels(
                                                                y_true=gold_truth_train, y_hat=mLLM_pred_train
                                                             ), model_name=args_.mLLM)
                                                            #  train_loader.get_result_values(train_loader.mLLM_results, key='binary_with_gold'),
                                                            #  helpers.get_binary_labels(
                                                            #      sim=train_loader.get_result_values(train_loader.mLLM_results, key='sim_with_gold')
                                                            #  ), 

                                                             
    evaluator.Evaluator().classification_metrics([1] * len(gold_truth_train), 
                                                 helpers.get_binary_labels(
                                                    y_true=mLLM_pred_train,
                                                    y_hat=sLLM_pred_train
                                                 ), 
                                                 model_name=args_.sLLM, oracle=args_.mLLM)
    
    evaluator.Evaluator().generation_metrics(gold_truth_train, sLLM_pred_train, model_name=args_.sLLM, 
                                            #  sim=train_loader.get_result_values(train_loader.sLLM_results, key='sim_with_gold')
                                             sim=helpers.get_binary_labels(
                                                 y_true=gold_truth_train,
                                                 y_hat=sLLM_pred_train,
                                                 return_sim=True)
                                             )
    sim_bound = evaluator.Evaluator().generation_metrics(gold_truth_train, mLLM_pred_train, model_name=args_.mLLM, 
                                                        #  sim=train_loader.get_result_values(train_loader.mLLM_results, 
                                                        #                                     key='sim_with_gold')
                                                        sim=helpers.get_binary_labels(
                                                            y_true=gold_truth_train,
                                                            y_hat=mLLM_pred_train,
                                                            return_sim=True)
                                                        )
    evaluator.Evaluator().generation_metrics(mLLM_pred_train, sLLM_pred_train, model_name=args_.sLLM, oracle=args_.mLLM,
                                             sim=helpers.get_binary_labels(
                                                 y_true=mLLM_pred_train,
                                                 y_hat=sLLM_pred_train,
                                                 return_sim=True)
                                            )
    
    args_.xi = acc_bound - 0.05   # override bound
    print(f'New bound: {args_.xi}, target error: {1. - args_.xi}')

    calibrator = None
    if args_.calibrate:
        probs = train_loader.obtain_generation_betas(quantile_based=True)
        calibrator = calibrators.SequenceConfidenceCalibrator()
        print(f'\nCalibrating confidence...')
        calibrator.fit(probs, 
                       helpers.get_binary_labels(y_true=gold_truth_train, 
                                                 y_hat=train_loader.get_result_values(train_loader.sLLM_results, key='pred'), 
                                                 return_sim=True))
        # calibrator.fit(probs, train_loader.get_result_values(train_loader.sLLM_results, key='binary_with_gold'))
        sLLM_betas_train = calibrator.predict(probs)
    else:
        sLLM_betas_train = train_loader.obtain_generation_betas(quantile_based=True)
    
    print(sLLM_betas_train)
    
    optimizer = optimizers.SingleThresholdOptimizer(y_true=[1] * len(gold_truth_train), 
                                                    # y_hat_s=train_loader.get_result_values(train_loader.sLLM_results, key='binary_with_gold'),
                                                    y_hat_s=helpers.get_binary_labels(y_true=gold_truth_train, y_hat=sLLM_pred_train), 
                                                    # y_hat_m=train_loader.get_result_values(train_loader.mLLM_results, key='binary_with_gold'),
                                                    y_hat_m=helpers.get_binary_labels(y_true=gold_truth_train, y_hat=mLLM_pred_train),
                                                    betas=sLLM_betas_train, xi=args_.xi, oracle=args_.oracle,
                                                    name='oracle' if args_.oracle else 'non_oracle',
                                                    cos_s=train_loader.get_result_values(train_loader.sLLM_results, key='sim_with_gold'),
                                                    cos_m=train_loader.get_result_values(train_loader.mLLM_results, key='sim_with_gold')
                                                    )
    
    opt_result = optimizer.optimize(c_s=sLLM_cost_param_aware, c_m=mLLM_cost_param_aware)
    tau_star, _ = opt_result
    
    eval_train = evaluator.Evaluator(
        y_true=gold_truth_train, y_hat_s=sLLM_pred_train, y_hat_m=mLLM_pred_train,
        betas=sLLM_betas_train, threshold=tau_star, xi=args_.xi, oracle=args_.oracle
    )

    print(f'\nStatistics for training set:')
    pprint.pprint(eval_train.evaluate_policy(
        c_s=sLLM_cost_param_aware, c_m=mLLM_cost_param_aware,
        mode='oracle' if args_.oracle else 'non_oracle'
    ))
    print()

    if args_.calculate_cost:
        cost_calc = cost_calculator.CostCalculator(results=test_loader.sLLM_results, model_id=helpers.get_cost_model(args_.sLLM))
        cost_calc.calculate()
        train_loader.sLLM_results = copy.deepcopy(cost_calc.results)
        del cost_calc

        cost_calc = cost_calculator.CostCalculator(results=test_loader.mLLM_results, model_id=helpers.get_cost_model(args_.mLLM))
        cost_calc.calculate()
        train_loader.mLLM_results = copy.deepcopy(cost_calc.results)
        del cost_calc

        sLLM_cost_test = helpers.cost_distribution_signature(test_loader.get_result_values(test_loader.sLLM_results,
                                                                                       key='prof_post_hoc_cost'))
        mLLM_cost_test = helpers.cost_distribution_signature(test_loader.get_result_values(test_loader.mLLM_results,
                                                                                           key='prof_post_hoc_cost'))
        print(f'\nCost signatures - test: sLLM = {sLLM_cost_test}, mLLM = {mLLM_cost_test}.\n')
        sLLM_cost_param_aware_test = helpers.cost_distribution_signature_param_aware(
            test_loader.get_result_values(test_loader.sLLM_results, key='prof_post_hoc_cost'), 
            parameter_count=args_.sLLM_param_count)
        mLLM_cost_param_aware_test = helpers.cost_distribution_signature_param_aware(
            test_loader.get_result_values(test_loader.mLLM_results, key='prof_post_hoc_cost'), 
            parameter_count=args_.mLLM_param_count)
        print(f'Cost signatures (param-aware) - test: sLLM = {sLLM_cost_param_aware_test},'
              f' mLLM = {mLLM_cost_param_aware_test}.\n')
    else:
        sLLM_cost_test = helpers.cost_distribution_signature(test_loader.get_result_values(test_loader.sLLM_results,
                                                                                           key='cost_tflops'))
        mLLM_cost_test = helpers.cost_distribution_signature(test_loader.get_result_values(test_loader.mLLM_results,
                                                                                           key='cost_tflops'))
        print(f'\nCost signatures - test: sLLM = {sLLM_cost_test}, mLLM = {mLLM_cost_test}.\n')
        sLLM_cost_param_aware_test = helpers.cost_distribution_signature_param_aware(
            test_loader.get_result_values(test_loader.sLLM_results, key='cost_tflops'), parameter_count=1)
        mLLM_cost_param_aware_test = helpers.cost_distribution_signature_param_aware(
            test_loader.get_result_values(test_loader.mLLM_results, key='cost_tflops'), parameter_count=4)
        print(f'Cost signatures (param-aware) - test: sLLM = {sLLM_cost_param_aware_test},'
              f' mLLM = {mLLM_cost_param_aware_test}.\n')
    
    

    sLLM_pred_test = test_loader.get_result_values(test_loader.sLLM_results, key='pred')
    mLLM_pred_test = test_loader.get_result_values(test_loader.mLLM_results, key='pred')
    if args_.calibrate:
        probs_test = test_loader.obtain_generation_betas(quantile_based=True)
        sLLM_betas_test = calibrator.predict(probs_test)
    else: sLLM_betas_test = test_loader.obtain_generation_betas(quantile_based=True)

    gold_truth_test = test_loader.get_result_values(test_loader.sLLM_results, key='gold_answer')
    
    # eval_ = evaluator.Evaluator()
    # evaluator.Evaluator().classification_metrics(gold_truth_test, sLLM_pred_test, model_name=args_.sLLM)
    # evaluator.Evaluator().classification_metrics(gold_truth_test, mLLM_pred_test, model_name=args_.mLLM)
    # evaluator.Evaluator().classification_metrics(mLLM_pred_test, sLLM_pred_test, 
    #                                              model_name=args_.sLLM, oracle=args_.mLLM)
    # del eval_

    eval_test = evaluator.Evaluator(
        y_true=gold_truth_test, y_hat_s=sLLM_pred_test, y_hat_m=mLLM_pred_test,
        betas=sLLM_betas_test, threshold=tau_star, xi=args_.xi, test=True, oracle=args_.oracle
    )

    print(f'\nStatistics for testing set:')
    test_eval = eval_test.evaluate_policy(
        c_s=sLLM_cost_param_aware_test, c_m=mLLM_cost_param_aware_test,
        mode='oracle' if args_.oracle else 'non_oracle',
        gold_truth=gold_truth_test if args_.oracle else None
    )
    pprint.pprint(test_eval)
    
    print()
    eval_test.threshold_plots_non_oracle(c_s=sLLM_cost_param_aware_test, c_m=mLLM_cost_param_aware_test,
                                         error_ind=test_eval['policy_global_error'], cost_ind=test_eval['policy_cost_avg'],
                                         gold_truth=gold_truth_test, rouge=False,
                                         filename_error=f"figures/{args.dataset}_{args.sLLM}_{args.mLLM}_xi{args.xi}_{"oracle" if args.oracle else "non_oracle"}_error_plot.pdf",
                                         filename_cost=f"figures/{args.dataset}_{args.sLLM}_{args.mLLM}_xi{args.xi}_{"oracle" if args.oracle else "non_oracle"}_cost_plot.pdf")

    return test_eval

if __name__ == '__main__':
    args = parser.parse_args()
    help_map = {
        action.dest: action.help
        for action in parser._actions
        if action.help is not None
    }
    print("Parsed arguments:")
    for key, value in vars(args).items():
        help_text = help_map.get(key, "no help available")
        print(f"  {key} ({help_text}): {value}")
    
    figures_directory = Path("figures/")
    figures_directory.mkdir(parents=True, exist_ok=True)

    result = run_single_threshold(args) if args.single_threshold else run_multi_thresholds(args)
    
    if args.save_results:
        results_directory = Path("results/")
        file_path = results_directory / f"{args.dataset}_{args.sLLM}_{args.mLLM}_xi{args.xi}_{"oracle" if args.oracle else "non_oracle"}.json"
        # Create the directory if it doesn't exist
        results_directory.mkdir(parents=True, exist_ok=True)

        # Save the dictionary to a JSON file
        with open(file_path, 'w') as json_file:
            json.dump(result, json_file, indent=4)

        print(f"Results saved to {file_path}")

    
