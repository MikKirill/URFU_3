# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе #2 выполнил:
- Кириллов Михаил Сергеевич
- XC21
Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 60 |
| Задание 2 | * | 20 |
| Задание 3 | * | 20 |


## Цель работы
Обучение виртуальной сферы посредством ML

## Задание 1
### Реализовать систему машинного обучения в связке Python - Google-Sheets – Unity. 1 Создим на сцене плоскость, шар и куб и изменим их цвета
![image](https://user-images.githubusercontent.com/94719239/197004628-8ac29194-46bf-4438-8e11-829e2b9c61e8.png)

## 2 Добавим скрипт RollerAgent.cs для сферы
```cs
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class RollerAgent : Agent
{
    Rigidbody rBody;
    // Start is called before the first frame update
    void Start()
    {
        rBody = GetComponent<Rigidbody>();
    }

    public Transform Target;
    public override void OnEpisodeBegin()
    {
        if (this.transform.localPosition.y < 0)
        {
            this.rBody.angularVelocity = Vector3.zero;
            this.rBody.velocity = Vector3.zero;
            this.transform.localPosition = new Vector3(0, 0.5f, 0);
        }

        Target.localPosition = new Vector3(Random.value * 8-4, 0.5f, Random.value * 8-4);
    }
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(Target.localPosition);
        sensor.AddObservation(this.transform.localPosition);
        sensor.AddObservation(rBody.velocity.x);
        sensor.AddObservation(rBody.velocity.z);
    }
    public float forceMultiplier = 10;
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = actionBuffers.ContinuousActions[0];
        controlSignal.z = actionBuffers.ContinuousActions[1];
        rBody.AddForce(controlSignal * forceMultiplier);

        float distanceToTarget = Vector3.Distance(this.transform.localPosition, Target.localPosition);

        if(distanceToTarget < 1.42f)
        {
            SetReward(1.0f);
            EndEpisode();
        }
        else if (this.transform.localPosition.y < 0)
        {
            EndEpisode();
        }
    }
}
```
3. Добавим в сферу Decision Requester и Behavior Parameters

4. В корень проекта добавим файл конфигурации нейронной сети для запуска ml-агента
![2](https://user-images.githubusercontent.com/94719239/197005835-b05b6eb4-94b6-40ce-b346-c7d509b00174.png)
5. Создаем несколько копий модели TargetArea, для ускорения их обучения и, соответственно, обучаем
![3](https://user-images.githubusercontent.com/94719239/197006806-56c537ba-7ecb-45bd-bdd6-9a6eb2f11a6f.png)
6. Все работает должным образом

## Задание 2
### Самостоятельно описать каждую строку файла конфигурации нейронной сети и найти необходимую информацию по компонентам Decision Requester, Behavior Parameters, добавленных сфере.
```cs
behaviors:
  RollerBall: # указываем id агента
    trainer_type: ppo # режим обучения (Proximal Policy Optimization)
    hyperparameters:
      batch_size: 10 # количество опытов на каждой итерации
      buffer_size: 100 # количество опыта, которое нужно набрать перед обновлением модели
      learning_rate: 3.0e-4 # начальная скорость обучения
      beta: 5.0e-4 # сила регуляции энтропии, увеличивает случайность действий
      epsilon: 0.2 # порог расхождений между старой и новой политиками при обновлении
      lambd: 0.99 # параметр регуляции, насколько сильно агент полагается на текущий value estimate
      num_epoch: 3 # количество проходов через буфер опыта, при выполнении оптимизации
      learning_rate_schedule: linear # определяет как скорость обучения изменяется с течением времени
                                     # linear линейно уменьшает скорость
    network_settings:
      normalize: false # отключаем нормализацию входных данных
      hidden_units: 128 # количество нейронов в скрытых слоях сети
      num_layers: 2 # количество скрытых слоёв в сети
    reward_signals:
      extrinsic:
        gamma: 0.99 # коэффициент скидки для будущих вознаграждений
        strength: 1.0 # коэффициент на который умножается вознаграждение
    max_steps: 500000 # общее количество шагов, которые должны быть выполнены в среде до завершения обучения
    time_horizon: 64 # сколько опыта нужно собрать для каждого агента, прежде чем добавлять его в буфер
    summary_freq: 10000 # количество опыта, который необходимо собрать перед созданием и отображением статистики
```
Decision Requester - запрашивает решение через регулярные промежутки времени.

Behavior Parameters - определяет принятие объектом решений, в него указывается какой тип поведения будет использоваться: уже обученная модель или удалённый процесс обучения.

## Задание 3
### Доработать сцену и обучить ML-Agent таким образом, чтобы шар перемещался между двумя кубами разного цвета. Кубы должны, как и впервом задании, случайно изменять кооринаты на плоскости.

1. Добавим второй куб и создадим для него цвет, например синий
![4Untitled](https://user-images.githubusercontent.com/94719239/197008292-dabb50ed-3e56-4ed8-9e7b-d47027b952f0.png)


